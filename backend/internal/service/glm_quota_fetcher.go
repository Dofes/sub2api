package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// GLMQuotaFetcher 从 GLM 监控 API 获取额度信息
type GLMQuotaFetcher struct {
	proxyRepo ProxyRepository
}

// NewGLMQuotaFetcher 创建 GLMQuotaFetcher
func NewGLMQuotaFetcher(proxyRepo ProxyRepository) *GLMQuotaFetcher {
	return &GLMQuotaFetcher{proxyRepo: proxyRepo}
}

// CanFetch 检查是否可以获取此账户的额度
func (f *GLMQuotaFetcher) CanFetch(account *Account) bool {
	if account.Platform != PlatformGLM {
		return false
	}
	apiKey := account.GetCredential("api_key")
	return apiKey != ""
}

// GLMQuotaLimitResponse GLM 配额限制 API 响应
type GLMQuotaLimitResponse struct {
	Data *GLMQuotaLimitData `json:"data"`
}

type GLMQuotaLimitData struct {
	Level  string          `json:"level"` // 账号等级: pro 等
	Limits []GLMQuotaLimit `json:"limits"`
}

// GLM 配额限制 unit 常量
const (
	glmUnitHours  = 3 // 小时窗口 (5h)
	glmUnitMonths = 5 // 月度窗口 (MCP)
	glmUnitWeeks  = 6 // 周窗口
)

type GLMQuotaLimit struct {
	Type          string          `json:"type"`          // TOKENS_LIMIT, TIME_LIMIT
	Unit          int             `json:"unit"`          // 3=小时, 5=月, 6=周
	Number        int             `json:"number"`        // 窗口数量 (e.g. 5小时, 1周)
	Percentage    float64         `json:"percentage"`    // 使用百分比 (0-100)
	CurrentValue  int64           `json:"currentValue"`  // 当前使用量
	Usage         int64           `json:"usage"`         // 总额度
	Remaining     int64           `json:"remaining"`     // 剩余额度
	NextResetTime int64           `json:"nextResetTime"` // 重置时间 (Unix毫秒时间戳)
	UsageDetails  json.RawMessage `json:"usageDetails"`  // TIME_LIMIT 详情
}

// FetchQuota 获取 GLM 账户配额信息
func (f *GLMQuotaFetcher) FetchQuota(ctx context.Context, account *Account, proxyURL string) (*UsageInfo, error) {
	apiKey := account.GetCredential("api_key")
	baseURL := f.getBaseURL(account)

	quotaURL := baseURL + "/api/monitor/usage/quota/limit"

	body, err := f.doRequest(ctx, quotaURL, apiKey, proxyURL)
	if err != nil {
		return nil, fmt.Errorf("fetch GLM quota failed: %w", err)
	}

	var resp GLMQuotaLimitResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse GLM quota response failed: %w", err)
	}

	return f.buildUsageInfo(resp.Data), nil
}

// getBaseURL 获取 GLM 监控 API 的基础 URL
// 从账号的 base_url 提取域名（去除 /api/anthropic 路径后缀）
func (f *GLMQuotaFetcher) getBaseURL(account *Account) string {
	baseURL := account.GetCredential("base_url")
	if baseURL == "" {
		return "https://open.bigmodel.cn"
	}
	// 解析 URL 提取 scheme + host
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "https://open.bigmodel.cn"
	}
	return parsed.Scheme + "://" + parsed.Host
}

// buildUsageInfo 将 GLM 配额限制数据转换为 UsageInfo
func (f *GLMQuotaFetcher) buildUsageInfo(data *GLMQuotaLimitData) *UsageInfo {
	now := time.Now()
	info := &UsageInfo{
		UpdatedAt: &now,
	}

	if data == nil {
		return info
	}

	for _, limit := range data.Limits {
		progress := &UsageProgress{
			Utilization: limit.Percentage,
		}

		// 解析 nextResetTime（毫秒时间戳）
		if limit.NextResetTime > 0 {
			resetTime := time.UnixMilli(limit.NextResetTime)
			progress.ResetsAt = &resetTime
			remaining := int(time.Until(resetTime).Seconds())
			if remaining < 0 {
				remaining = 0
			}
			progress.RemainingSeconds = remaining
		}

		switch limit.Type {
		case "TOKENS_LIMIT":
			switch limit.Unit {
			case glmUnitHours:
				// 5 小时 Token 窗口
				info.FiveHour = progress
			case glmUnitWeeks:
				// 每周 Token 窗口
				info.SevenDay = progress
			}
		}
		// TIME_LIMIT (月度 MCP) 暂不展示，对 API 转发场景无关
	}

	return info
}

// doRequest 执行 HTTP GET 请求
func (f *GLMQuotaFetcher) doRequest(ctx context.Context, apiURL, authToken, proxyURL string) ([]byte, error) {
	client := &http.Client{Timeout: 15 * time.Second}

	if proxyURL != "" {
		proxyParsed, err := url.Parse(proxyURL)
		if err == nil {
			client.Transport = &http.Transport{
				Proxy: http.ProxyURL(proxyParsed),
			}
		}
	}

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", authToken)
	req.Header.Set("Accept-Language", "en-US,en")
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GLM API returned status %d: %s", resp.StatusCode, string(body))
	}

	return body, nil
}

// GetProxyURL 获取账户的代理 URL
func (f *GLMQuotaFetcher) GetProxyURL(ctx context.Context, account *Account) string {
	if account.ProxyID == nil || f.proxyRepo == nil {
		return ""
	}
	proxy, err := f.proxyRepo.GetByID(ctx, *account.ProxyID)
	if err != nil || proxy == nil {
		return ""
	}
	return proxy.URL()
}
