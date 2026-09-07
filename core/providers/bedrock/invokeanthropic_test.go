package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// usesAnthropicInvokePath decides which Claude requests leave Converse for
// InvokeModel (#6825). Only a compact_20260112 edit on an Anthropic-family
// model qualifies; everything else must keep the Converse path it has today.
func TestUsesAnthropicInvokePath(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	compact := json.RawMessage(`{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":50000}}]}`)
	clearOnly := json.RawMessage(`{"edits":[{"type":"clear_tool_uses_20250919"}]}`)
	mixed := json.RawMessage(`{"edits":[{"type":"clear_tool_uses_20250919"},{"type":"compact_20260112"}]}`)

	tests := []struct {
		name  string
		model string
		cm    json.RawMessage
		want  bool
	}{
		{"claude with compaction edit", "us.anthropic.claude-sonnet-4-6", compact, true},
		{"claude with region prefix and compaction edit", "us-west-2/anthropic.claude-opus-4-6-v1", compact, true},
		{"claude with compaction among other edits", "anthropic.claude-sonnet-4-6", mixed, true},
		{"claude with clear_tool_uses only stays on converse", "anthropic.claude-sonnet-4-6", clearOnly, false},
		{"claude without context_management stays on converse", "anthropic.claude-sonnet-4-6", nil, false},
		{"claude with empty edits stays on converse", "anthropic.claude-sonnet-4-6", json.RawMessage(`{"edits":[]}`), false},
		{"nova with compaction edit stays on converse", "amazon.nova-pro-v1:0", compact, false},
		{"llama with compaction edit stays on converse", "meta.llama3-70b-instruct-v1:0", compact, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, usesAnthropicInvokePath(ctx, tt.model, tt.cm, nil))
		})
	}
}

func TestUsesAnthropicInvokePath_NilParams(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	require.False(t, chatUsesAnthropicInvokePath(ctx, &schemas.BifrostChatRequest{Model: "anthropic.claude-sonnet-4-6"}))
	require.False(t, responsesUsesAnthropicInvokePath(ctx, &schemas.BifrostResponsesRequest{Model: "anthropic.claude-sonnet-4-6"}))
	require.False(t, chatUsesAnthropicInvokePath(ctx, nil))
	require.False(t, responsesUsesAnthropicInvokePath(ctx, nil))
}

func TestInvokeURL_UsesRuntimeHostAndAction(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	provider := &BedrockProvider{}
	key := schemas.Key{BedrockKeyConfig: &schemas.BedrockKeyConfig{Region: schemas.NewSecretVar("us-east-1")}}

	url, region := provider.invokeURL(ctx, key, "us.anthropic.claude-sonnet-4-6", bedrockInvokeStreamAction)
	require.Equal(t, "us-east-1", region)
	require.Equal(t, "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke-with-response-stream", url)

	url, _ = provider.invokeURL(ctx, key, "eu-west-1/anthropic.claude-opus-4-6-v1", bedrockInvokeAction)
	require.Equal(t, "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-opus-4-6-v1/invoke", url)
}

// The /anthropic/v1/messages ingress (the path in #6825) does not populate the
// raw Params.ContextManagement field. AnthropicMessageRequest.ToBifrostResponsesRequest
// stores a typed *anthropic.ContextManagement under ExtraParams["context_management"],
// and older callers may store a plain map there. The routing predicate must see
// all three shapes, or the reporter's traffic never leaves Converse.
func TestUsesAnthropicInvokePath_ExtraParamsShapes(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	model := "us.anthropic.claude-opus-4-8"
	typed := &anthropic.ContextManagement{
		Edits: []anthropic.ContextManagementEdit{{Type: anthropic.ContextManagementEditTypeCompact}},
	}
	asMap := map[string]interface{}{
		"edits": []interface{}{map[string]interface{}{"type": "compact_20260112"}},
	}
	clearTyped := &anthropic.ContextManagement{
		Edits: []anthropic.ContextManagementEdit{{Type: anthropic.ContextManagementEditTypeClearToolUses}},
	}

	t.Run("responses typed pointer in extra params routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			ExtraParams: map[string]interface{}{"context_management": typed},
		}}
		require.True(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("responses map in extra params routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			ExtraParams: map[string]interface{}{"context_management": asMap},
		}}
		require.True(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("chat typed pointer in extra params routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostChatRequest{Model: model, Params: &schemas.ChatParameters{
			ExtraParams: map[string]interface{}{"context_management": typed},
		}}
		require.True(t, chatUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("clear-only typed pointer stays on converse", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			ExtraParams: map[string]interface{}{"context_management": clearTyped},
		}}
		require.False(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("raw field wins even when extra params is unrelated", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			ContextManagement: json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`),
			ExtraParams:       map[string]interface{}{"context_management": clearTyped},
		}}
		require.True(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
}

// The InvokeModel route (and Mantle) go through fasthttp, whose path
// normalisation decodes a percent-encoded inference-profile ARN in the model
// segment. The provider must build its fasthttp clients with that disabled;
// the streaming client is a clone, so the flag must survive cloning too.
func TestNewBedrockProvider_FasthttpClientsPreservePathEncoding(t *testing.T) {
	provider, err := NewBedrockProvider(&schemas.ProviderConfig{
		ConcurrencyAndBufferSize: schemas.ConcurrencyAndBufferSize{Concurrency: 1, BufferSize: 1},
	}, noopLogger{})
	require.NoError(t, err)
	require.True(t, provider.mantleClient.DisablePathNormalizing, "unary fasthttp client must preserve percent-encoded model paths")
	require.True(t, provider.mantleStreamingClient.DisablePathNormalizing, "streaming fasthttp client (a clone) must preserve percent-encoded model paths")
}

// Tool search is the second InvokeModel-only feature on Bedrock: "On Amazon
// Bedrock, server-side tool search is available only through the InvokeModel
// API, not the Converse API."
// (https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
// A request that carries a tool_search tool, or any tool with defer_loading
// (which only means something alongside tool search), must leave Converse.
func TestUsesAnthropicInvokePath_ToolSearch(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	model := "us.anthropic.claude-opus-4-6-v1"
	deferred := true

	t.Run("responses tool_search tool routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeToolSearch, Name: schemas.Ptr("tool_search_tool_regex")}},
		}}
		require.True(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("responses deferred function tool routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("get_weather"), DeferLoading: &deferred}},
		}}
		require.True(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("responses plain function tool stays on converse", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: model, Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("get_weather")}},
		}}
		require.False(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("chat dated tool_search type routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostChatRequest{Model: model, Params: &schemas.ChatParameters{
			Tools: []schemas.ChatTool{{Type: "tool_search_tool_regex_20251119"}},
		}}
		require.True(t, chatUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("chat deferred function tool routes to invoke", func(t *testing.T) {
		req := &schemas.BifrostChatRequest{Model: model, Params: &schemas.ChatParameters{
			Tools: []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "get_weather"}, DeferLoading: &deferred}},
		}}
		require.True(t, chatUsesAnthropicInvokePath(ctx, req))
	})
	t.Run("nova with tool_search stays on converse", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Model: "amazon.nova-pro-v1:0", Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeToolSearch, Name: schemas.Ptr("tool_search_tool_regex")}},
		}}
		require.False(t, responsesUsesAnthropicInvokePath(ctx, req))
	})
}

// CountTokens stays on the Converse count-tokens envelope for ordinary
// requests, but a request that is routed to InvokeModel must be counted with
// the same body it will be sent with. AWS's CountTokens input is a union of
// "converse" and "invokeModel" ({"body": <base64 of the InvokeModel body>}):
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokensInput.html
func TestBedrockCountTokensBody_UsesInvokeModelInputForRoutedRequests(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	provider := &BedrockProvider{}
	role := schemas.ResponsesInputMessageRoleUser
	text := "Hello!"
	input := []schemas.ResponsesMessage{{Role: &role, Content: &schemas.ResponsesMessageContent{ContentStr: &text}}}

	t.Run("compaction request counts via invokeModel", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Provider: schemas.Bedrock, Model: "global.anthropic.claude-sonnet-4-6", Input: input, Params: &schemas.ResponsesParameters{
			ContextManagement: json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`),
		}}
		body, err := provider.buildCountTokensBody(ctx, req)
		require.NoError(t, err)
		encoded := gjson.GetBytes(body, "input.invokeModel.body").String()
		require.NotEmpty(t, encoded, "expected input.invokeModel.body, got %s", string(body))
		require.False(t, gjson.GetBytes(body, "input.converse").Exists(), "union must carry one member: %s", string(body))
		decoded, decErr := base64.StdEncoding.DecodeString(encoded)
		require.NoError(t, decErr)
		require.Equal(t, "bedrock-2023-05-31", gjson.GetBytes(decoded, "anthropic_version").String(), "decoded body: %s", string(decoded))
		require.Equal(t, "compact_20260112", gjson.GetBytes(decoded, "context_management.edits.0.type").String())
	})
	t.Run("tool_search request counts via invokeModel", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Provider: schemas.Bedrock, Model: "global.anthropic.claude-sonnet-4-6", Input: input, Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeToolSearch, Name: schemas.Ptr("tool_search_tool_regex")}},
		}}
		body, err := provider.buildCountTokensBody(ctx, req)
		require.NoError(t, err)
		require.True(t, gjson.GetBytes(body, "input.invokeModel.body").Exists(), "expected invokeModel input, got %s", string(body))
	})
	t.Run("plain request counts via converse", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{Provider: schemas.Bedrock, Model: "global.anthropic.claude-sonnet-4-6", Input: input}
		body, err := provider.buildCountTokensBody(ctx, req)
		require.NoError(t, err)
		require.True(t, gjson.GetBytes(body, "input.converse").Exists(), "expected converse input, got %s", string(body))
		require.False(t, gjson.GetBytes(body, "input.invokeModel").Exists())
	})
}
