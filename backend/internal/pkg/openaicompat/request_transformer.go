package openaicompat

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
)

// TransformClaudeToOpenAI 将 Claude Messages API 请求转换为 OpenAI Chat Completions 格式
func TransformClaudeToOpenAI(claudeReq *antigravity.ClaudeRequest) ([]byte, error) {
	req := ChatRequest{
		Model:       claudeReq.Model,
		MaxTokens:   claudeReq.MaxTokens,
		Temperature: claudeReq.Temperature,
		TopP:        claudeReq.TopP,
		Stream:      claudeReq.Stream,
	}

	// 流式请求需要 include_usage 来获取 token 用量
	if claudeReq.Stream {
		req.StreamOptions = &StreamOpts{IncludeUsage: true}
	}

	// 转换 thinking → reasoning
	if claudeReq.Thinking != nil && (claudeReq.Thinking.Type == "enabled" || claudeReq.Thinking.Type == "adaptive") {
		effort := "high"
		if claudeReq.Thinking.BudgetTokens > 0 && claudeReq.Thinking.BudgetTokens <= 4096 {
			effort = "low"
		} else if claudeReq.Thinking.BudgetTokens > 4096 && claudeReq.Thinking.BudgetTokens <= 16384 {
			effort = "medium"
		}
		req.Reasoning = &ReasoningConfig{Effort: effort}
	}

	// 转换 system prompt
	var messages []ChatMessage
	systemMsg, err := buildSystemMessage(claudeReq.System)
	if err != nil {
		return nil, fmt.Errorf("build system message: %w", err)
	}
	if systemMsg != nil {
		messages = append(messages, *systemMsg)
	}

	// 转换 messages
	for i, msg := range claudeReq.Messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("convert message %d: %w", i, err)
		}
		messages = append(messages, converted...)
	}

	req.Messages = messages

	// 转换 tools
	if len(claudeReq.Tools) > 0 {
		req.Tools = convertTools(claudeReq.Tools)
	}

	// 转换 tool_choice
	if len(claudeReq.ToolChoice) > 0 {
		req.ToolChoice = convertToolChoice(claudeReq.ToolChoice)
	}

	return json.Marshal(req)
}

// buildSystemMessage 将 Claude system prompt 转换为 OpenAI system message
func buildSystemMessage(system json.RawMessage) (*ChatMessage, error) {
	if len(system) == 0 {
		return nil, nil
	}

	// 尝试解析为字符串
	var sysStr string
	if err := json.Unmarshal(system, &sysStr); err == nil {
		if strings.TrimSpace(sysStr) == "" {
			return nil, nil
		}
		content, _ := json.Marshal(sysStr)
		return &ChatMessage{Role: "system", Content: content}, nil
	}

	// 尝试解析为 SystemBlock 数组
	var sysBlocks []antigravity.SystemBlock
	if err := json.Unmarshal(system, &sysBlocks); err == nil {
		var texts []string
		for _, block := range sysBlocks {
			if block.Type == "text" && strings.TrimSpace(block.Text) != "" {
				texts = append(texts, block.Text)
			}
		}
		if len(texts) == 0 {
			return nil, nil
		}
		combined := strings.Join(texts, "\n\n")
		content, _ := json.Marshal(combined)
		return &ChatMessage{Role: "system", Content: content}, nil
	}

	return nil, nil
}

// convertMessage 将单条 Claude 消息转换为 OpenAI 消息（可能拆分为多条）
func convertMessage(msg antigravity.ClaudeMessage) ([]ChatMessage, error) {
	// 尝试解析 content 为字符串
	var textContent string
	if err := json.Unmarshal(msg.Content, &textContent); err == nil {
		content, _ := json.Marshal(textContent)
		return []ChatMessage{{Role: msg.Role, Content: content}}, nil
	}

	// 解析为内容块数组
	var blocks []antigravity.ContentBlock
	if err := json.Unmarshal(msg.Content, &blocks); err != nil {
		// 无法解析，原样传递
		return []ChatMessage{{Role: msg.Role, Content: msg.Content}}, nil
	}

	if msg.Role == "assistant" {
		return convertAssistantBlocks(blocks)
	}

	return convertUserBlocks(msg.Role, blocks)
}

// convertUserBlocks 转换 user 角色的内容块
func convertUserBlocks(role string, blocks []antigravity.ContentBlock) ([]ChatMessage, error) {
	var messages []ChatMessage
	var contentParts []ContentPart

	for _, block := range blocks {
		switch block.Type {
		case "text":
			contentParts = append(contentParts, ContentPart{
				Type: "text",
				Text: block.Text,
			})

		case "image":
			if block.Source != nil && block.Source.Type == "base64" {
				dataURL := fmt.Sprintf("data:%s;base64,%s", block.Source.MediaType, block.Source.Data)
				contentParts = append(contentParts, ContentPart{
					Type:     "image_url",
					ImageURL: &ImageURL{URL: dataURL},
				})
			}

		case "tool_result":
			// tool_result 需要作为独立的 tool message
			// 先把之前积累的 content 输出
			if len(contentParts) > 0 {
				partsJSON, _ := json.Marshal(contentParts)
				messages = append(messages, ChatMessage{Role: role, Content: partsJSON})
				contentParts = nil
			}

			// 提取 tool result 内容
			resultText := extractToolResultText(block)
			content, _ := json.Marshal(resultText)
			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    content,
				ToolCallID: block.ToolUseID,
			})

		case "thinking":
			// 跳过 thinking blocks
			continue
		}
	}

	// 输出剩余的 content parts
	if len(contentParts) > 0 {
		if len(contentParts) == 1 && contentParts[0].Type == "text" {
			// 单个文本，简化为字符串
			content, _ := json.Marshal(contentParts[0].Text)
			messages = append(messages, ChatMessage{Role: role, Content: content})
		} else {
			partsJSON, _ := json.Marshal(contentParts)
			messages = append(messages, ChatMessage{Role: role, Content: partsJSON})
		}
	}

	return messages, nil
}

// convertAssistantBlocks 转换 assistant 角色的内容块
func convertAssistantBlocks(blocks []antigravity.ContentBlock) ([]ChatMessage, error) {
	var textParts []string
	var toolCalls []ToolCall
	var thinkingParts []string
	var lastSignature string

	for _, block := range blocks {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)

		case "tool_use":
			argsJSON, err := json.Marshal(block.Input)
			if err != nil {
				argsJSON = []byte("{}")
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      block.Name,
					Arguments: string(argsJSON),
				},
			})

		case "thinking":
			// 保留 thinking 内容和 signature
			if block.Thinking != "" {
				thinkingParts = append(thinkingParts, block.Thinking)
			}
			if block.Signature != "" {
				lastSignature = block.Signature
			}
		}
	}

	msg := ChatMessage{Role: "assistant"}

	// 将历史消息中的 thinking 内容和 signature 一起传递
	if len(thinkingParts) > 0 {
		msg.ThinkingField = &ThinkingField{
			Content:   strings.Join(thinkingParts, "\n"),
			Signature: lastSignature,
		}
	}

	if len(textParts) > 0 {
		combined := strings.Join(textParts, "")
		content, _ := json.Marshal(combined)
		msg.Content = content
	}

	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return []ChatMessage{msg}, nil
}

// extractToolResultText 从 tool_result 块提取文本内容
func extractToolResultText(block antigravity.ContentBlock) string {
	if len(block.Content) == 0 {
		if block.IsError {
			return "Tool execution failed."
		}
		return "Command executed successfully."
	}

	// 尝试解析为字符串
	var str string
	if err := json.Unmarshal(block.Content, &str); err == nil {
		if strings.TrimSpace(str) == "" {
			if block.IsError {
				return "Tool execution failed."
			}
			return "Command executed successfully."
		}
		return str
	}

	// 尝试解析为数组
	var arr []map[string]any
	if err := json.Unmarshal(block.Content, &arr); err == nil {
		var texts []string
		for _, item := range arr {
			if text, ok := item["text"].(string); ok {
				texts = append(texts, text)
			}
		}
		result := strings.Join(texts, "\n")
		if strings.TrimSpace(result) != "" {
			return result
		}
	}

	// 返回原始 JSON
	return string(block.Content)
}

// convertTools 将 Claude 工具定义转换为 OpenAI function 格式
func convertTools(claudeTools []antigravity.ClaudeTool) []Tool {
	var tools []Tool
	for _, ct := range claudeTools {
		name := strings.TrimSpace(ct.Name)
		if name == "" {
			continue
		}

		var description string
		var parameters map[string]any

		if ct.Type == "custom" && ct.Custom != nil {
			description = ct.Custom.Description
			parameters = ct.Custom.InputSchema
		} else {
			description = ct.Description
			parameters = ct.InputSchema
		}

		// web_search 等特殊工具类型跳过
		if ct.Type == "web_search_20250305" || ct.Type == "web_search" {
			continue
		}

		if parameters == nil {
			parameters = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}

		tools = append(tools, Tool{
			Type: "function",
			Function: FunctionDef{
				Name:        name,
				Description: description,
				Parameters:  parameters,
			},
		})
	}
	return tools
}

// convertToolChoice 将 Claude tool_choice 转换为 OpenAI tool_choice
// Claude 格式: {"type": "auto"} / {"type": "any"} / {"type": "tool", "name": "xxx"}
// OpenAI 格式: "auto" / "required" / "none" / {"type": "function", "function": {"name": "xxx"}}
func convertToolChoice(raw json.RawMessage) any {
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name,omitempty"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil
	}
	switch tc.Type {
	case "auto":
		return "auto"
	case "any":
		// Claude "any" = 必须调用某个工具 = OpenAI "required"
		return "required"
	case "none":
		return "none"
	case "tool":
		// 指定调用某个具体工具
		return map[string]any{
			"type":     "function",
			"function": map[string]string{"name": tc.Name},
		}
	default:
		return nil
	}
}
