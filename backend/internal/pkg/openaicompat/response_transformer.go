package openaicompat

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
)

// TransformOpenAIToClaude 将 OpenAI Chat Completions 响应转换为 Claude Messages API 格式
func TransformOpenAIToClaude(body []byte, originalModel string) ([]byte, *antigravity.ClaudeUsage, error) {
	var resp ChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, nil, fmt.Errorf("parse openai response: %w", err)
	}

	// 构建 Claude content blocks
	var content []antigravity.ClaudeContentItem
	var hasToolUse bool

	if len(resp.Choices) > 0 {
		msg := resp.Choices[0].Message

		// Reasoning → Claude thinking block
		// 支持 reasoning、reasoning_content 和 thinking 三种字段名
		reasoning := msg.Reasoning
		if reasoning == "" {
			reasoning = msg.ReasoningContent
		}
		// 某些上游用 thinking 字段（带 signature）
		var thinkingSignature string
		if msg.ThinkingField != nil && msg.ThinkingField.Content != "" {
			reasoning = msg.ThinkingField.Content
			thinkingSignature = msg.ThinkingField.Signature
		}
		if reasoning != "" {
			// 如果上游没返回 signature，生成一个假签名（Claude Code 多轮对话需要）
			if thinkingSignature == "" {
				thinkingSignature = generateFakeSignature()
			}
			content = append(content, antigravity.ClaudeContentItem{
				Type:      "thinking",
				Thinking:  reasoning,
				Signature: thinkingSignature,
			})
		}

		// 文本内容
		var textContent string
		if len(msg.Content) > 0 {
			_ = json.Unmarshal(msg.Content, &textContent)
		}
		if textContent != "" {
			content = append(content, antigravity.ClaudeContentItem{
				Type: "text",
				Text: textContent,
			})
		}

		// Tool calls
		for _, tc := range msg.ToolCalls {
			hasToolUse = true

			var input any
			if tc.Function.Arguments != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
			}
			if input == nil {
				input = map[string]any{}
			}

			content = append(content, antigravity.ClaudeContentItem{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	}

	// 如果没有任何内容，添加空文本块
	if len(content) == 0 {
		content = append(content, antigravity.ClaudeContentItem{
			Type: "text",
			Text: "",
		})
	}

	// 转换 finish_reason → stop_reason
	stopReason := "end_turn"
	if len(resp.Choices) > 0 {
		stopReason = mapFinishReason(resp.Choices[0].FinishReason, hasToolUse)
	}

	// 提取 usage
	usage := extractUsage(resp.Usage)

	// 构建 Claude 响应
	claudeResp := antigravity.ClaudeResponse{
		ID:         convertID(resp.ID),
		Type:       "message",
		Role:       "assistant",
		Model:      originalModel,
		Content:    content,
		StopReason: stopReason,
		Usage:      *usage,
	}

	respBytes, err := json.Marshal(claudeResp)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal claude response: %w", err)
	}

	return respBytes, usage, nil
}

// mapFinishReason 将 OpenAI finish_reason 映射为 Claude stop_reason
func mapFinishReason(finishReason string, hasToolUse bool) string {
	if hasToolUse {
		return "tool_use"
	}
	switch finishReason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "content_filter":
		return "end_turn"
	default:
		return "end_turn"
	}
}

// extractUsage 从 OpenAI usage 提取 Claude usage
// input_tokens = prompt_tokens - cached_tokens，因为 Claude 的 input_tokens 不含 cached 部分
func extractUsage(u *Usage) *antigravity.ClaudeUsage {
	usage := &antigravity.ClaudeUsage{}
	if u == nil {
		return usage
	}
	cachedTokens := 0
	if u.PromptTokensDetails != nil {
		cachedTokens = u.PromptTokensDetails.CachedTokens
	}
	usage.InputTokens = u.PromptTokens - cachedTokens
	if usage.InputTokens < 0 {
		usage.InputTokens = 0
	}
	usage.OutputTokens = u.CompletionTokens
	usage.CacheReadInputTokens = cachedTokens
	return usage
}

// convertID 转换响应 ID 格式
func convertID(openaiID string) string {
	if openaiID != "" {
		return openaiID
	}
	return "msg_" + randomID()
}

func randomID() string {
	const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, 12)
	for i := range b {
		b[i] = chars[rand.Intn(len(chars))]
	}
	return string(b)
}

// generateFakeSignature 为不返回 signature 的上游（DeepSeek、GLM 等）生成假签名
// Claude Code 多轮对话时需要 thinking block 包含 signature
func generateFakeSignature() string {
	return strconv.FormatInt(time.Now().UnixMilli(), 10)
}

// TransformOpenAIErrorToClaude 将 OpenAI 格式错误转换为 Claude 格式错误
func TransformOpenAIErrorToClaude(body []byte, statusCode int) []byte {
	var openaiErr ErrorResponse
	if err := json.Unmarshal(body, &openaiErr); err != nil || openaiErr.Error == nil {
		// 无法解析，原样返回
		return body
	}

	// 映射错误类型
	errType := mapErrorType(statusCode)

	claudeErr := antigravity.ClaudeError{
		Type: "error",
		Error: antigravity.ErrorDetail{
			Type:    errType,
			Message: openaiErr.Error.Message,
		},
	}

	result, err := json.Marshal(claudeErr)
	if err != nil {
		return body
	}
	return result
}

// mapErrorType 根据 HTTP 状态码映射 Claude 错误类型
func mapErrorType(statusCode int) string {
	switch {
	case statusCode == 400:
		return "invalid_request_error"
	case statusCode == 401:
		return "authentication_error"
	case statusCode == 403:
		return "permission_error"
	case statusCode == 404:
		return "not_found_error"
	case statusCode == 429:
		return "rate_limit_error"
	case statusCode >= 500:
		return "api_error"
	default:
		return "api_error"
	}
}

// ExtractUsageFromBody 从非流式响应体提取 usage（用于 service 层）
func ExtractUsageFromBody(body []byte) *antigravity.ClaudeUsage {
	var resp ChatResponse
	if json.Unmarshal(body, &resp) != nil {
		return &antigravity.ClaudeUsage{}
	}
	return extractUsage(resp.Usage)
}

// IsOpenAIErrorFormat 检测响应体是否为 OpenAI 错误格式
func IsOpenAIErrorFormat(body []byte) bool {
	return strings.Contains(string(body), `"error"`) && strings.Contains(string(body), `"message"`)
}
