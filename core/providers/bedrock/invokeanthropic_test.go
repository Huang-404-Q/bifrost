package bedrock

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
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
