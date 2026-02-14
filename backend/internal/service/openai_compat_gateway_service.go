package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openaicompat"
	"github.com/gin-gonic/gin"
)

// OpenAICompatGatewayService 处理 OpenAI 兼容平台的请求转发
// 将 Claude Messages API 请求转换为 OpenAI Chat Completions 格式后发送到任意 OpenAI 兼容 API
// 支持 OpenRouter、LiteLLM、One API、vLLM 等所有 OpenAI Chat Completions 兼容的上游
type OpenAICompatGatewayService struct {
	httpUpstream   HTTPUpstream
	settingService *SettingService
}

// NewOpenAICompatGatewayService 创建 OpenAICompatGatewayService
func NewOpenAICompatGatewayService(
	httpUpstream HTTPUpstream,
	settingService *SettingService,
) *OpenAICompatGatewayService {
	return &OpenAICompatGatewayService{
		httpUpstream:   httpUpstream,
		settingService: settingService,
	}
}

// Forward 转发请求到 OpenAI 兼容上游
// 接收 Claude Messages API 格式请求，转换为 OpenAI Chat Completions 格式后发送
func (s *OpenAICompatGatewayService) Forward(ctx context.Context, c *gin.Context, account *Account, body []byte) (*ForwardResult, error) {
	startTime := time.Now()

	// 获取上游配置
	baseURL := strings.TrimSpace(account.GetCredential("base_url"))
	apiKey := strings.TrimSpace(account.GetCredential("api_key"))
	if baseURL == "" || apiKey == "" {
		return nil, fmt.Errorf("openai-compat account missing base_url or api_key")
	}
	baseURL = strings.TrimSuffix(baseURL, "/")
	upstreamURL := baseURL + "/chat/completions"

	// 解析 Claude 请求
	var claudeReq antigravity.ClaudeRequest
	if err := json.Unmarshal(body, &claudeReq); err != nil {
		return nil, fmt.Errorf("parse claude request: %w", err)
	}
	if strings.TrimSpace(claudeReq.Model) == "" {
		return nil, fmt.Errorf("missing model")
	}
	originalModel := claudeReq.Model
	billingModel := originalModel

	// 模型映射
	if mappedModel := account.GetMappedModel(originalModel); mappedModel != "" && mappedModel != originalModel {
		claudeReq.Model = mappedModel
		billingModel = mappedModel
	}

	// 转换为 OpenAI Chat Completions 格式
	openaiBody, err := openaicompat.TransformClaudeToOpenAI(&claudeReq)
	if err != nil {
		return nil, fmt.Errorf("transform request: %w", err)
	}

	// 创建请求
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(openaiBody))
	if err != nil {
		return nil, fmt.Errorf("create upstream request: %w", err)
	}

	// 设置请求头
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// 代理 URL
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	// 发送请求
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		log.Printf("[OpenAICompat] upstream request failed: %v", err)
		return nil, fmt.Errorf("upstream request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// 处理错误响应
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))

		// 429 时返回 UpstreamFailoverError 以触发账号切换
		if resp.StatusCode == http.StatusTooManyRequests {
			return nil, &UpstreamFailoverError{
				StatusCode:   resp.StatusCode,
				ResponseBody: respBody,
			}
		}

		// 转换错误格式：OpenAI → Claude
		claudeErrBody := openaicompat.TransformOpenAIErrorToClaude(respBody, resp.StatusCode)
		c.Header("Content-Type", "application/json")
		c.Status(resp.StatusCode)
		_, _ = c.Writer.Write(claudeErrBody)

		return &ForwardResult{
			Model: billingModel,
		}, nil
	}

	// 处理成功响应
	var usage *ClaudeUsage
	var firstTokenMs *int
	var clientDisconnect bool

	if claudeReq.Stream {
		streamRes := s.streamResponse(c, resp, startTime, originalModel)
		usage = streamRes.usage
		firstTokenMs = streamRes.firstTokenMs
		clientDisconnect = streamRes.clientDisconnect
	} else {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("read upstream response: %w", err)
		}

		// 某些上游可能用 HTTP 200 包装错误（错误码在 JSON body 内部）
		var errResp openaicompat.ErrorResponse
		if json.Unmarshal(respBody, &errResp) == nil && errResp.Error != nil {
			statusCode := http.StatusBadGateway
			if code, ok := errResp.Error.Code.(float64); ok {
				statusCode = int(code)
			}
			if statusCode == http.StatusTooManyRequests {
				return nil, &UpstreamFailoverError{
					StatusCode:   statusCode,
					ResponseBody: respBody,
				}
			}
			claudeErrBody := openaicompat.TransformOpenAIErrorToClaude(respBody, statusCode)
			c.Header("Content-Type", "application/json")
			c.Status(statusCode)
			_, _ = c.Writer.Write(claudeErrBody)
			return &ForwardResult{Model: billingModel}, nil
		}

		// 转换响应：OpenAI → Claude
		claudeRespBody, respUsage, err := openaicompat.TransformOpenAIToClaude(respBody, originalModel)
		if err != nil {
			// 转换失败，透传原始响应
			log.Printf("[OpenAICompat] transform response failed: %v, passing through", err)
			c.Header("Content-Type", resp.Header.Get("Content-Type"))
			c.Status(http.StatusOK)
			_, _ = c.Writer.Write(respBody)
			usage = &ClaudeUsage{}
		} else {
			c.Header("Content-Type", "application/json")
			c.Status(http.StatusOK)
			_, _ = c.Writer.Write(claudeRespBody)
			usage = &ClaudeUsage{
				InputTokens:          respUsage.InputTokens,
				OutputTokens:         respUsage.OutputTokens,
				CacheReadInputTokens: respUsage.CacheReadInputTokens,
			}
		}
	}

	duration := time.Since(startTime)
	log.Printf("[OpenAICompat] status=success model=%s duration_ms=%d", billingModel, duration.Milliseconds())

	return &ForwardResult{
		Model:            billingModel,
		Stream:           claudeReq.Stream,
		Duration:         duration,
		FirstTokenMs:     firstTokenMs,
		ClientDisconnect: clientDisconnect,
		Usage: ClaudeUsage{
			InputTokens:          usage.InputTokens,
			OutputTokens:         usage.OutputTokens,
			CacheReadInputTokens: usage.CacheReadInputTokens,
		},
	}, nil
}

// openaiCompatStreamResult 流式响应结果
type openaiCompatStreamResult struct {
	usage            *ClaudeUsage
	firstTokenMs     *int
	clientDisconnect bool
}

// streamResponse 处理流式响应：将 OpenAI SSE 转换为 Claude SSE 后写回客户端
func (s *OpenAICompatGatewayService) streamResponse(c *gin.Context, resp *http.Response, startTime time.Time, originalModel string) *openaiCompatStreamResult {
	processor := openaicompat.NewStreamingProcessor(originalModel)

	scanner := bufio.NewScanner(resp.Body)
	maxLineSize := defaultMaxLineSize
	if s.settingService.cfg != nil && s.settingService.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.settingService.cfg.Gateway.MaxLineSize
	}
	scanner.Buffer(make([]byte, 64*1024), maxLineSize)

	type scanEvent struct {
		line string
		err  error
	}
	events := make(chan scanEvent, 16)
	done := make(chan struct{})
	sendEvent := func(ev scanEvent) bool {
		select {
		case events <- ev:
			return true
		case <-done:
			return false
		}
	}
	var lastReadAt int64
	atomic.StoreInt64(&lastReadAt, time.Now().UnixNano())
	go func() {
		defer close(events)
		for scanner.Scan() {
			atomic.StoreInt64(&lastReadAt, time.Now().UnixNano())
			if !sendEvent(scanEvent{line: scanner.Text()}) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			_ = sendEvent(scanEvent{err: err})
		}
	}()
	defer close(done)

	streamInterval := time.Duration(0)
	if s.settingService.cfg != nil && s.settingService.cfg.Gateway.StreamDataIntervalTimeout > 0 {
		streamInterval = time.Duration(s.settingService.cfg.Gateway.StreamDataIntervalTimeout) * time.Second
	}
	var intervalTicker *time.Ticker
	if streamInterval > 0 {
		intervalTicker = time.NewTicker(streamInterval)
		defer intervalTicker.Stop()
	}
	var intervalCh <-chan time.Time
	if intervalTicker != nil {
		intervalCh = intervalTicker.C
	}

	// 设置 SSE 响应头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	c.Status(http.StatusOK)

	flusher, _ := c.Writer.(http.Flusher)
	cw := newAntigravityClientWriter(c.Writer, flusher, "openaicompat")

	var firstTokenMs *int

	for {
		select {
		case ev, ok := <-events:
			if !ok {
				// 流结束，发送最终事件
				finalData, finalUsage := processor.Finish()
				if len(finalData) > 0 {
					cw.Write(finalData)
				}
				usage := &ClaudeUsage{
					InputTokens:          finalUsage.InputTokens,
					OutputTokens:         finalUsage.OutputTokens,
					CacheReadInputTokens: finalUsage.CacheReadInputTokens,
				}
				return &openaiCompatStreamResult{usage: usage, firstTokenMs: firstTokenMs, clientDisconnect: cw.Disconnected()}
			}
			if ev.err != nil {
				if disconnect, handled := handleStreamReadError(ev.err, cw.Disconnected(), "openaicompat"); handled {
					_, finalUsage := processor.Finish()
					usage := &ClaudeUsage{
						InputTokens:          finalUsage.InputTokens,
						OutputTokens:         finalUsage.OutputTokens,
						CacheReadInputTokens: finalUsage.CacheReadInputTokens,
					}
					return &openaiCompatStreamResult{usage: usage, firstTokenMs: firstTokenMs, clientDisconnect: disconnect}
				}
				log.Printf("[OpenAICompat] Stream read error: %v", ev.err)
				_, finalUsage := processor.Finish()
				usage := &ClaudeUsage{
					InputTokens:          finalUsage.InputTokens,
					OutputTokens:         finalUsage.OutputTokens,
					CacheReadInputTokens: finalUsage.CacheReadInputTokens,
				}
				return &openaiCompatStreamResult{usage: usage, firstTokenMs: firstTokenMs}
			}

			line := ev.line

			// 记录首 token 时间
			if firstTokenMs == nil && len(line) > 0 {
				ms := int(time.Since(startTime).Milliseconds())
				firstTokenMs = &ms
			}

			// 转换 OpenAI SSE → Claude SSE
			claudeEvents := processor.ProcessLine(line)
			if len(claudeEvents) > 0 {
				cw.Write(claudeEvents)
			}

		case <-intervalCh:
			lastRead := time.Unix(0, atomic.LoadInt64(&lastReadAt))
			if time.Since(lastRead) < streamInterval {
				continue
			}
			if cw.Disconnected() {
				log.Printf("[OpenAICompat] Upstream timeout after client disconnect, returning collected usage")
				_, finalUsage := processor.Finish()
				usage := &ClaudeUsage{
					InputTokens:          finalUsage.InputTokens,
					OutputTokens:         finalUsage.OutputTokens,
					CacheReadInputTokens: finalUsage.CacheReadInputTokens,
				}
				return &openaiCompatStreamResult{usage: usage, firstTokenMs: firstTokenMs, clientDisconnect: true}
			}
			log.Printf("[OpenAICompat] Stream data interval timeout")
			_, finalUsage := processor.Finish()
			usage := &ClaudeUsage{
				InputTokens:          finalUsage.InputTokens,
				OutputTokens:         finalUsage.OutputTokens,
				CacheReadInputTokens: finalUsage.CacheReadInputTokens,
			}
			return &openaiCompatStreamResult{usage: usage, firstTokenMs: firstTokenMs}
		}
	}
}

// TestConnection 测试 OpenAI 兼容账号连接（非流式）
func (s *OpenAICompatGatewayService) TestConnection(ctx context.Context, account *Account, modelID string) (*TestConnectionResult, error) {
	// 获取凭据
	baseURL := strings.TrimSpace(account.GetCredential("base_url"))
	apiKey := strings.TrimSpace(account.GetCredential("api_key"))
	if baseURL == "" || apiKey == "" {
		return nil, fmt.Errorf("openai-compat account missing base_url or api_key")
	}
	baseURL = strings.TrimSuffix(baseURL, "/")
	upstreamURL := baseURL + "/chat/completions"

	// 模型映射
	mappedModel := modelID
	if m := account.GetMappedModel(modelID); m != "" && m != modelID {
		mappedModel = m
	}

	// 构建 OpenAI Chat Completions 请求
	chatReq := openaicompat.ChatRequest{
		Model: mappedModel,
		Messages: []openaicompat.ChatMessage{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
		MaxTokens: 16,
		Stream:    false,
	}
	reqBody, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// 代理 URL
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	// 发送请求
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API returned %d: %s", resp.StatusCode, string(respBody))
	}

	log.Printf("[OpenAICompat] TestConnection raw response: %s", string(respBody))

	// 某些上游可能用 HTTP 200 包装错误（错误码在 JSON body 内部）
	var errResp openaicompat.ErrorResponse
	if json.Unmarshal(respBody, &errResp) == nil && errResp.Error != nil {
		return nil, fmt.Errorf("upstream error (code %v): %s", errResp.Error.Code, errResp.Error.Message)
	}

	// 解析响应
	var chatResp openaicompat.ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	text := ""
	if len(chatResp.Choices) > 0 {
		msg := chatResp.Choices[0].Message
		raw := msg.Content
		// Content 可能是 JSON 字符串 "hello" 或纯文本
		var s string
		if json.Unmarshal(raw, &s) == nil {
			text = s
		} else {
			text = string(raw)
		}
		// 某些模型（如 reasoning 模型）把输出放在 reasoning 字段而非 content
		if text == "" && msg.Reasoning != "" {
			text = msg.Reasoning
		}
		if text == "" && msg.ReasoningContent != "" {
			text = msg.ReasoningContent
		}
	}

	return &TestConnectionResult{
		Text:        text,
		MappedModel: mappedModel,
	}, nil
}
