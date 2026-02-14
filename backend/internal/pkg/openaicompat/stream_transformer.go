package openaicompat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
)

// StreamingProcessor 将 OpenAI SSE 流转换为 Claude SSE 流
type StreamingProcessor struct {
	originalModel    string
	messageStartSent bool
	messageStopSent  bool
	blockIndex       int
	blockOpen        bool // 当前是否有未关闭的 content block
	blockType        string
	usedTool         bool
	thinkingStarted  bool // 是否已开始 thinking block
	thinkingGotSig   bool // 是否收到过真实 signature

	// 工具调用状态：追踪多个并发 tool_calls
	activeToolCalls map[int]*toolCallState

	// 累计 usage
	usage antigravity.ClaudeUsage
}

// toolCallState 追踪单个 tool call 的增量构建
type toolCallState struct {
	ID        string
	Name      string
	Arguments strings.Builder
	Started   bool // 是否已发送 content_block_start
}

// NewStreamingProcessor 创建流式处理器
func NewStreamingProcessor(originalModel string) *StreamingProcessor {
	return &StreamingProcessor{
		originalModel:   originalModel,
		activeToolCalls: make(map[int]*toolCallState),
	}
}

// ProcessLine 处理一行 SSE 数据，返回转换后的 Claude SSE 事件
func (p *StreamingProcessor) ProcessLine(line string) []byte {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil
	}

	// 处理 data: [DONE]
	if line == "data: [DONE]" || line == "data:[DONE]" {
		return p.finishIfNeeded()
	}

	// 只处理 data: 行
	if !strings.HasPrefix(line, "data:") {
		return nil
	}

	data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if data == "" || data == "[DONE]" {
		return p.finishIfNeeded()
	}

	var chunk StreamChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil
	}

	var result bytes.Buffer

	// 首次处理：发送 message_start
	if !p.messageStartSent {
		result.Write(p.emitMessageStart(chunk.ID))
	}

	// 更新 usage（在最后一个 chunk 中包含 usage）
	// input_tokens = prompt_tokens - cached_tokens
	if chunk.Usage != nil {
		cachedTokens := 0
		if chunk.Usage.PromptTokensDetails != nil {
			cachedTokens = chunk.Usage.PromptTokensDetails.CachedTokens
		}
		p.usage.InputTokens = chunk.Usage.PromptTokens - cachedTokens
		if p.usage.InputTokens < 0 {
			p.usage.InputTokens = 0
		}
		p.usage.OutputTokens = chunk.Usage.CompletionTokens
		p.usage.CacheReadInputTokens = cachedTokens
	}

	// 处理 choices
	for _, choice := range chunk.Choices {
		delta := choice.Delta

		// 处理 thinking/reasoning 内容
		if delta.Thinking != nil {
			if delta.Thinking.Content != "" {
				result.Write(p.processThinkingDelta(delta.Thinking.Content))
			}
			if delta.Thinking.Signature != "" {
				result.Write(p.processSignatureDelta(delta.Thinking.Signature))
			}
		}

		// 处理 reasoning_content / reasoning 字段 (不同提供商格式)
		if delta.ReasoningContent != "" {
			result.Write(p.processThinkingDelta(delta.ReasoningContent))
		} else if delta.Reasoning != "" {
			result.Write(p.processThinkingDelta(delta.Reasoning))
		}

		// 处理文本内容
		if delta.Content != "" {
			result.Write(p.processTextDelta(delta.Content))
		}

		// 处理 tool calls
		if len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				result.Write(p.processToolCallDelta(tc))
			}
		}

		// 处理 finish_reason
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			result.Write(p.emitFinish(*choice.FinishReason))
		}
	}

	return result.Bytes()
}

// Finish 结束处理，返回最终事件和用量
func (p *StreamingProcessor) Finish() ([]byte, *antigravity.ClaudeUsage) {
	var result bytes.Buffer
	if !p.messageStopSent {
		result.Write(p.emitFinish("stop"))
	}
	return result.Bytes(), &p.usage
}

// emitMessageStart 发送 message_start 事件
func (p *StreamingProcessor) emitMessageStart(responseID string) []byte {
	if p.messageStartSent {
		return nil
	}

	if responseID == "" {
		responseID = "msg_" + randomID()
	}

	message := map[string]any{
		"id":            responseID,
		"type":          "message",
		"role":          "assistant",
		"content":       []any{},
		"model":         p.originalModel,
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": antigravity.ClaudeUsage{
			InputTokens:  p.usage.InputTokens,
			OutputTokens: 0,
		},
	}

	event := map[string]any{
		"type":    "message_start",
		"message": message,
	}

	p.messageStartSent = true
	return formatSSE("message_start", event)
}

// processTextDelta 处理文本增量
func (p *StreamingProcessor) processTextDelta(text string) []byte {
	var result bytes.Buffer

	// 如果当前有非 text 的 block，先关闭
	// thinking block 需要注入假签名
	if p.blockOpen && p.blockType != "text" {
		if p.blockType == "thinking" {
			result.Write(p.closeThinkingWithFakeSignature())
		} else {
			result.Write(p.closeBlock())
		}
	}

	// 如果没有打开的 text block，开一个
	if !p.blockOpen {
		result.Write(p.openBlock("text", map[string]any{
			"type": "text",
			"text": "",
		}))
	}

	// 发送 text delta
	delta := map[string]any{
		"type": "text_delta",
		"text": text,
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": p.blockIndex,
		"delta": delta,
	}
	result.Write(formatSSE("content_block_delta", event))

	return result.Bytes()
}

// processThinkingDelta 处理 thinking/reasoning 增量
func (p *StreamingProcessor) processThinkingDelta(text string) []byte {
	var result bytes.Buffer

	// 如果当前有非 thinking 的 block，先关闭
	if p.blockOpen && p.blockType != "thinking" {
		result.Write(p.closeBlock())
	}

	// 如果没有打开的 thinking block，开一个
	if !p.blockOpen {
		result.Write(p.openBlock("thinking", map[string]any{
			"type":     "thinking",
			"thinking": "",
		}))
		p.thinkingStarted = true
	}

	// 发送 thinking delta
	delta := map[string]any{
		"type":     "thinking_delta",
		"thinking": text,
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": p.blockIndex,
		"delta": delta,
	}
	result.Write(formatSSE("content_block_delta", event))

	return result.Bytes()
}

// processSignatureDelta 处理 thinking signature
func (p *StreamingProcessor) processSignatureDelta(signature string) []byte {
	var result bytes.Buffer

	// signature 应该在 thinking block 内
	if !p.blockOpen || p.blockType != "thinking" {
		return nil
	}

	p.thinkingGotSig = true

	delta := map[string]any{
		"type":      "signature_delta",
		"signature": signature,
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": p.blockIndex,
		"delta": delta,
	}
	result.Write(formatSSE("content_block_delta", event))

	// signature 发送后关闭 thinking block
	result.Write(p.closeBlock())

	return result.Bytes()
}

// closeThinkingWithFakeSignature 在 thinking block 未收到真实 signature 时注入假签名并关闭
// 适用于 DeepSeek/GLM 等不返回 signature 的上游
func (p *StreamingProcessor) closeThinkingWithFakeSignature() []byte {
	if !p.blockOpen || p.blockType != "thinking" {
		return nil
	}
	if p.thinkingGotSig {
		// 已有真实 signature，正常关闭
		return p.closeBlock()
	}

	var result bytes.Buffer

	// 注入假签名
	fakeSig := strconv.FormatInt(time.Now().UnixMilli(), 10)
	delta := map[string]any{
		"type":      "signature_delta",
		"signature": fakeSig,
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": p.blockIndex,
		"delta": delta,
	}
	result.Write(formatSSE("content_block_delta", event))
	result.Write(p.closeBlock())

	return result.Bytes()
}

// processToolCallDelta 处理工具调用增量
func (p *StreamingProcessor) processToolCallDelta(tc ToolCall) []byte {
	var result bytes.Buffer
	p.usedTool = true

	// 使用 OpenAI 的 index 字段来区分多个并发 tool_calls
	idx := tc.Index

	state, exists := p.activeToolCalls[idx]

	if tc.ID != "" && !exists {
		// 新 tool call 开始
		// 如果 thinking block 未关闭，先注入假签名并关闭
		if p.blockOpen && p.blockType == "thinking" {
			result.Write(p.closeThinkingWithFakeSignature())
		} else if p.blockOpen {
			result.Write(p.closeBlock())
		}

		// ID fallback: 某些上游可能不返回 ID
		toolID := tc.ID
		if toolID == "" {
			toolID = fmt.Sprintf("call_%d_%d", time.Now().UnixMilli(), idx)
		}
		toolName := tc.Function.Name
		if toolName == "" {
			toolName = fmt.Sprintf("tool_%d", idx)
		}

		state = &toolCallState{
			ID:   toolID,
			Name: toolName,
		}
		p.activeToolCalls[idx] = state

		// 发送 content_block_start
		toolUseBlock := map[string]any{
			"type":  "tool_use",
			"id":    toolID,
			"name":  toolName,
			"input": map[string]any{},
		}
		result.Write(p.openBlock("tool_use", toolUseBlock))
		state.Started = true
	} else if !exists {
		// arguments 数据但无 state，创建 fallback
		if p.blockOpen && p.blockType == "thinking" {
			result.Write(p.closeThinkingWithFakeSignature())
		} else if p.blockOpen {
			result.Write(p.closeBlock())
		}

		toolID := fmt.Sprintf("call_%d_%d", time.Now().UnixMilli(), idx)
		toolName := tc.Function.Name
		if toolName == "" {
			toolName = fmt.Sprintf("tool_%d", idx)
		}

		state = &toolCallState{ID: toolID, Name: toolName}
		p.activeToolCalls[idx] = state

		toolUseBlock := map[string]any{
			"type":  "tool_use",
			"id":    toolID,
			"name":  toolName,
			"input": map[string]any{},
		}
		result.Write(p.openBlock("tool_use", toolUseBlock))
		state.Started = true
	}

	// 累积 arguments
	if tc.Function.Arguments != "" && state != nil {
		state.Arguments.WriteString(tc.Function.Arguments)

		// 发送 input_json_delta
		delta := map[string]any{
			"type":         "input_json_delta",
			"partial_json": tc.Function.Arguments,
		}
		event := map[string]any{
			"type":  "content_block_delta",
			"index": p.blockIndex,
			"delta": delta,
		}
		result.Write(formatSSE("content_block_delta", event))
	}

	return result.Bytes()
}

// emitFinish 发送结束事件
func (p *StreamingProcessor) emitFinish(finishReason string) []byte {
	if p.messageStopSent {
		return nil
	}

	var result bytes.Buffer

	// 关闭当前 block（thinking block 需要注入假签名）
	if p.blockOpen {
		if p.blockType == "thinking" {
			result.Write(p.closeThinkingWithFakeSignature())
		} else {
			result.Write(p.closeBlock())
		}
	}

	// 确定 stop_reason
	stopReason := "end_turn"
	if p.usedTool {
		stopReason = "tool_use"
	} else {
		switch finishReason {
		case "stop":
			stopReason = "end_turn"
		case "tool_calls":
			stopReason = "tool_use"
		case "length":
			stopReason = "max_tokens"
		}
	}

	// message_delta
	deltaEvent := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": p.usage.OutputTokens,
		},
	}
	result.Write(formatSSE("message_delta", deltaEvent))

	// message_stop
	stopEvent := map[string]any{
		"type": "message_stop",
	}
	result.Write(formatSSE("message_stop", stopEvent))

	p.messageStopSent = true
	return result.Bytes()
}

// finishIfNeeded 在 [DONE] 时补发结束事件
func (p *StreamingProcessor) finishIfNeeded() []byte {
	if p.messageStopSent {
		return nil
	}
	if !p.messageStartSent {
		return nil
	}
	return p.emitFinish("stop")
}

// openBlock 开始新的 content block
func (p *StreamingProcessor) openBlock(blockType string, contentBlock map[string]any) []byte {
	if p.blockOpen {
		return nil // 不应该在未关闭时打开新 block
	}

	event := map[string]any{
		"type":          "content_block_start",
		"index":         p.blockIndex,
		"content_block": contentBlock,
	}

	p.blockOpen = true
	p.blockType = blockType
	return formatSSE("content_block_start", event)
}

// closeBlock 关闭当前 content block
func (p *StreamingProcessor) closeBlock() []byte {
	if !p.blockOpen {
		return nil
	}

	event := map[string]any{
		"type":  "content_block_stop",
		"index": p.blockIndex,
	}

	p.blockOpen = false
	p.blockIndex++
	p.blockType = ""
	return formatSSE("content_block_stop", event)
}

// formatSSE 格式化 SSE 事件
func formatSSE(eventType string, data any) []byte {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil
	}
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(jsonData)))
}
