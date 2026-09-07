package anthropic

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Regression tests for maximhq/bifrost#6825.
//
// The Bedrock provider routes Claude requests that carry a compact_20260112
// edit to InvokeModel / InvokeModelWithResponseStream, because AWS documents
// compaction as unsupported on Converse:
// https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-compaction.html
//
// InvokeModel takes the native Anthropic Messages body with three Bedrock
// specifics, per
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html:
//   - anthropic_version must be "bedrock-2023-05-31"
//   - the model is in the URL, so the body carries no "model"
//   - streaming is selected by the URL, so the body carries no "stream"
//   - beta features are opted into via the anthropic_beta body array
// The shared anthropic request builder must produce exactly that shape when
// cfg.Provider is schemas.Bedrock.

const bedrockInvokeCompactionContextManagement = `{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":50000}}]}`

func assertBedrockInvokeBodyShape(t *testing.T, body []byte) {
	t.Helper()
	if providerUtils.JSONFieldExists(body, "model") {
		t.Errorf("InvokeModel body must not carry model (it is in the URL), got: %s", string(body))
	}
	if providerUtils.JSONFieldExists(body, "stream") {
		t.Errorf("InvokeModel body must not carry stream (the URL selects streaming), got: %s", string(body))
	}
	if got := providerUtils.GetJSONField(body, "anthropic_version").String(); got != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version = %q, want %q", got, "bedrock-2023-05-31")
	}
	betas := providerUtils.GetJSONField(body, "anthropic_beta")
	if !betas.Exists() || !betas.IsArray() {
		t.Fatalf("anthropic_beta array missing, got: %s", string(body))
	}
	var betaValues []string
	for _, b := range betas.Array() {
		betaValues = append(betaValues, b.String())
	}
	if !slices.Contains(betaValues, AnthropicCompactionBetaHeader) {
		t.Errorf("anthropic_beta = %v, want it to contain %q", betaValues, AnthropicCompactionBetaHeader)
	}
	if got := providerUtils.GetJSONField(body, "context_management.edits.0.type").String(); got != string(ContextManagementEditTypeCompact) {
		t.Errorf("context_management.edits.0.type = %q, want %q; body=%s", got, ContextManagementEditTypeCompact, string(body))
	}
}

func TestBuildAnthropicResponsesRequestBody_BedrockInvokeShape(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.Bedrock,
		Model:    "us.anthropic.claude-sonnet-4-6",
		Input:    makeSimpleInput("Hello!"),
		Params: &schemas.ResponsesParameters{
			ContextManagement: json.RawMessage(bedrockInvokeCompactionContextManagement),
		},
	}
	body, err := BuildAnthropicResponsesRequestBody(ctx, request, AnthropicRequestBuildConfig{
		Provider:    schemas.Bedrock,
		Model:       "us.anthropic.claude-sonnet-4-6",
		IsStreaming: true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertBedrockInvokeBodyShape(t, body)
}

func TestBuildAnthropicChatRequestBody_BedrockInvokeShape(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock,
		Model:    "us.anthropic.claude-sonnet-4-6",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello!")}}},
		Params: &schemas.ChatParameters{
			ContextManagement: json.RawMessage(bedrockInvokeCompactionContextManagement),
		},
	}
	body, err := BuildAnthropicChatRequestBody(ctx, request, AnthropicRequestBuildConfig{
		Provider:    schemas.Bedrock,
		Model:       "us.anthropic.claude-sonnet-4-6",
		IsStreaming: true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertBedrockInvokeBodyShape(t, body)
}

// Tool search is InvokeModel-only on Bedrock (see the routing tests in the
// bedrock package). Once a request is routed there, the shared builder must keep
// the tool_search tool, keep defer_loading on the deferred function tool, and
// opt in with the tool-search-tool-2025-10-19 beta in the anthropic_beta array.
func TestBuildAnthropicResponsesRequestBody_BedrockInvokeKeepsToolSearch(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.Bedrock,
		Model:    "us.anthropic.claude-sonnet-4-6",
		Input:    makeSimpleInput("What is the weather in Paris?"),
		Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{
				responsesToolFromJSON(t, `{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"}`),
				responsesToolFromJSON(t, `{"type":"function","name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]},"defer_loading":true}`),
			},
		},
	}
	body, err := BuildAnthropicResponsesRequestBody(ctx, request, AnthropicRequestBuildConfig{
		Provider:      schemas.Bedrock,
		Model:         "us.anthropic.claude-sonnet-4-6",
		ValidateTools: true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tools := providerUtils.GetJSONField(body, "tools").Array()
	var sawToolSearch, sawDeferred bool
	for _, tool := range tools {
		if strings.HasPrefix(tool.Get("type").String(), "tool_search_tool_") {
			sawToolSearch = true
		}
		if tool.Get("name").String() == "get_weather" && tool.Get("defer_loading").Bool() {
			sawDeferred = true
		}
	}
	if !sawToolSearch {
		t.Errorf("tool_search tool was stripped from the InvokeModel body: %s", string(body))
	}
	if !sawDeferred {
		t.Errorf("defer_loading was stripped from the deferred function tool: %s", string(body))
	}
	var betas []string
	for _, b := range providerUtils.GetJSONField(body, "anthropic_beta").Array() {
		betas = append(betas, b.String())
	}
	if !slices.Contains(betas, AnthropicToolSearchBetaHeader) {
		t.Errorf("anthropic_beta = %v, want it to contain %q", betas, AnthropicToolSearchBetaHeader)
	}
}
