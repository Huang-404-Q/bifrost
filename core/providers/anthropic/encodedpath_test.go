package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// Regression tests for maximhq/bifrost#6825 (InvokeModel routing on Bedrock).
//
// Bedrock model paths may carry a percent-encoded inference-profile ARN, e.g.
// /model/arn%3Aaws%3Abedrock%3A...%3Aapplication-inference-profile%2Fabc%2Fglobal.anthropic.claude-sonnet-4-6/invoke.
// net/http (the Converse path) sends that path verbatim. fasthttp, which the
// shared anthropic handlers use, normalises the path on parse and re-quotes it
// on write, so the ARN's %2F and %3A reach AWS as literal "/" and ":" and the
// request lands on a route AWS does not have (UnknownOperationException). The
// handlers must send the caller's escaping unchanged whenever the client asks
// for it (fasthttp.Client.DisablePathNormalizing), including through the
// streaming and large-response client clones the handlers build per request.

const encodedARNModel = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123/global.anthropic.claude-sonnet-4-6"

type recordedRequestURI struct {
	mu  sync.Mutex
	uri string
}

func (r *recordedRequestURI) set(v string) { r.mu.Lock(); r.uri = v; r.mu.Unlock() }
func (r *recordedRequestURI) get() string  { r.mu.Lock(); defer r.mu.Unlock(); return r.uri }

func encodedPathServer(t *testing.T, rec *recordedRequestURI, streaming bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// r.RequestURI is the request-target exactly as it arrived on the wire.
		rec.set(r.RequestURI)
		if streaming {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(anthropicMessageStart + anthropicTextDelta + anthropicMessageStop))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
}

func assertEncodedPathPreserved(t *testing.T, got string) {
	t.Helper()
	want := "/model/" + url.PathEscape(encodedARNModel) + "/invoke"
	if !strings.HasPrefix(got, want) {
		t.Errorf("wire request-target lost the caller's percent-encoding\n got:  %s\n want: %s", got, want)
	}
}

func TestHandleAnthropicResponsesRequest_PreservesEncodedPath(t *testing.T) {
	rec := &recordedRequestURI{}
	server := encodedPathServer(t, rec, false)
	defer server.Close()

	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.Bedrock,
		Model:    "global.anthropic.claude-sonnet-4-6",
		Input:    makeSimpleInput("Hello!"),
	}
	requestURL := server.URL + "/model/" + url.PathEscape(encodedARNModel) + "/invoke"

	_, bifrostErr := HandleAnthropicResponsesRequest(ctx, &fasthttp.Client{DisablePathNormalizing: true}, requestURL, request,
		AnthropicRequestBuildConfig{Provider: schemas.Bedrock, Model: "global.anthropic.claude-sonnet-4-6"},
		map[string]string{}, nil, nil, truncationTestLogger{})
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %s", bifrostErr.Error.Message)
	}
	assertEncodedPathPreserved(t, rec.get())
}

func TestHandleAnthropicResponsesStream_PreservesEncodedPath(t *testing.T) {
	rec := &recordedRequestURI{}
	server := encodedPathServer(t, rec, true)
	defer server.Close()

	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.Bedrock,
		Model:    "global.anthropic.claude-sonnet-4-6",
		Input:    makeSimpleInput("Hello!"),
	}
	jsonData, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, request, AnthropicRequestBuildConfig{
		Provider: schemas.Bedrock, Model: "global.anthropic.claude-sonnet-4-6", IsStreaming: true,
	})
	if bifrostErr != nil {
		t.Fatalf("build: %s", bifrostErr.Error.Message)
	}
	requestURL := server.URL + "/model/" + url.PathEscape(encodedARNModel) + "/invoke-with-response-stream"

	stream, bifrostErr := HandleAnthropicResponsesStream(ctx, &fasthttp.Client{DisablePathNormalizing: true}, requestURL, jsonData,
		map[string]string{}, nil, 30, nil, false, false, schemas.Bedrock,
		truncationPassthroughPostHook, nil, nil, truncationTestLogger{}, nil)
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %s", bifrostErr.Error.Message)
	}
	collectTruncationChunks(t, stream)
	assertEncodedPathPreserved(t, rec.get())
}

func TestHandleAnthropicChatCompletionStreaming_PreservesEncodedPath(t *testing.T) {
	rec := &recordedRequestURI{}
	server := encodedPathServer(t, rec, true)
	defer server.Close()

	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock,
		Model:    "global.anthropic.claude-sonnet-4-6",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello!")}}},
	}
	jsonData, bifrostErr := BuildAnthropicChatRequestBody(ctx, request, AnthropicRequestBuildConfig{
		Provider: schemas.Bedrock, Model: "global.anthropic.claude-sonnet-4-6", IsStreaming: true,
	})
	if bifrostErr != nil {
		t.Fatalf("build: %s", bifrostErr.Error.Message)
	}
	requestURL := server.URL + "/model/" + url.PathEscape(encodedARNModel) + "/invoke-with-response-stream"

	stream, bifrostErr := HandleAnthropicChatCompletionStreaming(ctx, &fasthttp.Client{DisablePathNormalizing: true}, requestURL, jsonData,
		map[string]string{}, nil, 30, nil, false, false, schemas.Bedrock,
		truncationPassthroughPostHook, nil, nil, truncationTestLogger{}, nil)
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %s", bifrostErr.Error.Message)
	}
	collectTruncationChunks(t, stream)
	assertEncodedPathPreserved(t, rec.get())
}
