package openaicompat

import "encoding/json"

// OpenAI Chat Completions 请求/响应类型定义
// 适用于所有 OpenAI Chat Completions 兼容 API（OpenRouter、LiteLLM、One API、vLLM 等）

// ChatRequest OpenAI Chat Completions 请求
type ChatRequest struct {
	Model         string           `json:"model"`
	Messages      []ChatMessage    `json:"messages"`
	MaxTokens     int              `json:"max_tokens,omitempty"`
	Temperature   *float64         `json:"temperature,omitempty"`
	TopP          *float64         `json:"top_p,omitempty"`
	Stream        bool             `json:"stream,omitempty"`
	Tools         []Tool           `json:"tools,omitempty"`
	ToolChoice    any              `json:"tool_choice,omitempty"` // string ("auto"/"none"/"required") 或 object
	StreamOptions *StreamOpts      `json:"stream_options,omitempty"`
	Reasoning     *ReasoningConfig `json:"reasoning,omitempty"`
}

// StreamOpts 流式选项
type StreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

// ReasoningConfig reasoning 配置 (对应 Claude 的 thinking)
type ReasoningConfig struct {
	Effort string `json:"effort,omitempty"` // "high", "medium", "low"
}

// ThinkingField 用于在历史消息中传递 thinking+signature
type ThinkingField struct {
	Content   string `json:"content,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// ChatMessage OpenAI 消息
type ChatMessage struct {
	Role             string            `json:"role"` // system, user, assistant, tool
	Content          json.RawMessage   `json:"content,omitempty"`
	Reasoning        string            `json:"reasoning,omitempty"`
	ReasoningContent string            `json:"reasoning_content,omitempty"` // 部分模型使用此字段
	ReasoningDetails []ReasoningDetail `json:"reasoning_details,omitempty"`
	ThinkingField    *ThinkingField    `json:"thinking,omitempty"` // 带 signature 的 thinking 传递
	ToolCalls        []ToolCall        `json:"tool_calls,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
	Name             string            `json:"name,omitempty"`
}

// ReasoningDetail reasoning 详情
type ReasoningDetail struct {
	Type string `json:"type,omitempty"` // "reasoning.text", "reasoning.signature" 等
	Text string `json:"text,omitempty"`
}

// ToolCall 工具调用
type ToolCall struct {
	Index    int          `json:"index,omitempty"` // 流式时标识 tool call 顺序
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall 函数调用
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// Tool 工具定义
type Tool struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// FunctionDef 函数定义
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ChatResponse OpenAI 非流式响应
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   *Usage       `json:"usage,omitempty"`
}

// ChatChoice 选择项
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"` // stop, tool_calls, length
}

// StreamChunk OpenAI 流式 chunk
type StreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Model   string              `json:"model"`
	Choices []StreamChunkChoice `json:"choices"`
	Usage   *Usage              `json:"usage,omitempty"`
}

// StreamChunkChoice 流式选择项
type StreamChunkChoice struct {
	Index        int              `json:"index"`
	Delta        StreamChunkDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason"` // nil or "stop", "tool_calls", "length"
}

// StreamChunkDelta 流式增量
type StreamChunkDelta struct {
	Role             string         `json:"role,omitempty"`
	Content          string         `json:"content,omitempty"`
	Thinking         *ThinkingDelta `json:"thinking,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"` // 部分模型使用此字段
	Reasoning        string         `json:"reasoning,omitempty"`         // 部分模型使用此字段
	ToolCalls        []ToolCall     `json:"tool_calls,omitempty"`
}

// ThinkingDelta reasoning/thinking 流式增量
type ThinkingDelta struct {
	Content   string `json:"content,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// Usage 用量
type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// PromptTokensDetails prompt token 详情
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// ContentPart OpenAI 多模态内容块
type ContentPart struct {
	Type     string    `json:"type"` // "text", "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL 图片 URL
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ErrorResponse OpenAI 错误响应
type ErrorResponse struct {
	Error *ErrorDetail `json:"error,omitempty"`
}

// ErrorDetail 错误详情
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code,omitempty"` // string or int
}
