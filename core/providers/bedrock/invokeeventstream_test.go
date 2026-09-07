package bedrock

import (
	"bytes"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// writeInvokeChunk frames one native Anthropic SSE event the way
// InvokeModelWithResponseStream does: a "chunk" event whose payload is
// {"bytes": base64(event JSON)}.
func writeInvokeChunk(t *testing.T, w io.Writer, eventJSON string) {
	t.Helper()
	payload := []byte(`{"bytes":"` + base64.StdEncoding.EncodeToString([]byte(eventJSON)) + `"}`)
	writeEventStreamEvent(t, w, "chunk", payload)
}

func TestInvokeEventStreamReader_YieldsAnthropicEvents(t *testing.T) {
	var buf bytes.Buffer
	writeInvokeChunk(t, &buf, `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`)
	writeInvokeChunk(t, &buf, `{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":""}}`)
	writeInvokeChunk(t, &buf, `{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"summary"}}`)

	reader := newInvokeEventStreamReader(&buf)

	eventType, data, err := reader.ReadEvent()
	require.NoError(t, err)
	require.Equal(t, "message_start", eventType)
	require.Contains(t, string(data), `"id":"msg_1"`)

	eventType, data, err = reader.ReadEvent()
	require.NoError(t, err)
	require.Equal(t, "content_block_start", eventType)
	require.Contains(t, string(data), `"type":"compaction"`)

	eventType, data, err = reader.ReadEvent()
	require.NoError(t, err)
	require.Equal(t, "content_block_delta", eventType)
	require.Contains(t, string(data), `"compaction_delta"`)

	_, _, err = reader.ReadEvent()
	require.ErrorIs(t, err, io.EOF)
}

func TestInvokeEventStreamReader_ExceptionFrameIsAnError(t *testing.T) {
	var buf bytes.Buffer
	enc := eventstream.NewEncoder()
	headers := eventstream.Headers{
		{Name: ":message-type", Value: eventstream.StringValue("exception")},
		{Name: ":exception-type", Value: eventstream.StringValue("validationException")},
		{Name: ":content-type", Value: eventstream.StringValue("application/json")},
	}
	require.NoError(t, enc.Encode(&buf, eventstream.Message{
		Headers: headers,
		Payload: []byte(`{"message":"compaction trigger must be at least 50000 tokens"}`),
	}))

	reader := newInvokeEventStreamReader(&buf)
	_, _, err := reader.ReadEvent()
	require.Error(t, err)
	require.Contains(t, err.Error(), "compaction trigger must be at least 50000 tokens")

	// The error must carry the classified BifrostError so the shared stream
	// loop can forward it unchanged: validationException is terminal.
	var carrier providerUtils.BifrostErrorCarrier
	require.ErrorAs(t, err, &carrier)
	typed := carrier.BifrostError()
	require.NotNil(t, typed)
	assert.True(t, typed.IsBifrostError, "validationException is not retryable")
	require.NotNil(t, typed.Type)
	assert.Equal(t, "validationException", *typed.Type)
}

func TestInvokeEventStreamReader_RetryableExceptionKeepsClassification(t *testing.T) {
	var buf bytes.Buffer
	writeEventStreamException(t, &buf, "throttlingException", "slow down")

	_, _, err := newInvokeEventStreamReader(&buf).ReadEvent()
	var carrier providerUtils.BifrostErrorCarrier
	require.ErrorAs(t, err, &carrier)
	typed := carrier.BifrostError()
	require.NotNil(t, typed)
	assert.False(t, typed.IsBifrostError, "throttlingException must stay retryable")
	require.NotNil(t, typed.StatusCode)
	assert.Equal(t, 429, *typed.StatusCode)
}

func TestInvokeEventStreamReader_SkipsEmptyChunks(t *testing.T) {
	var buf bytes.Buffer
	writeEventStreamEvent(t, &buf, "chunk", []byte(`{"bytes":""}`))
	writeInvokeChunk(t, &buf, `{"type":"message_stop"}`)

	reader := newInvokeEventStreamReader(&buf)
	eventType, _, err := reader.ReadEvent()
	require.NoError(t, err)
	require.Equal(t, "message_stop", eventType)
}

// newTestInvokeStreamServer serves one AWS event-stream exception frame on any
// request, over TLS because the InvokeModel URL is always https, and points the
// provider's fasthttp streaming client at it.
func newTestInvokeStreamServer(t *testing.T, excType, msg string) (*BedrockProvider, func()) {
	t.Helper()
	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		writeEventStreamException(t, w, excType, msg)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	provider := newTestProviderWithServer(t, ts)
	addr := strings.TrimPrefix(ts.URL, "https://")
	provider.mantleStreamingClient = &fasthttp.Client{
		Dial:                   func(string) (net.Conn, error) { return net.Dial("tcp", addr) },
		TLSConfig:              &tls.Config{InsecureSkipVerify: true},
		DisablePathNormalizing: true,
		ReadTimeout:            5 * time.Second,
		WriteTimeout:           5 * time.Second,
	}
	return provider, ts.Close
}

// A retryable AWS exception delivered as the first InvokeModelWithResponseStream
// frame must reach the caller with the same classification the Converse path
// gives it (IsBifrostError:false plus the mapped status), so the retry gate in
// executeRequestWithRetries can act on it. Mirrors
// TestChatCompletionStream_RetryableException_ChunkIsRetryable for the invoke route.
func TestInvokeAnthropicResponsesStream_RetryableException_ChunkIsRetryable(t *testing.T) {
	tests := []struct {
		excType        string
		expectedStatus int
	}{
		{"throttlingException", 429},
		{"serviceUnavailableException", 503},
	}
	for _, tc := range tests {
		t.Run(tc.excType, func(t *testing.T) {
			provider, closeServer := newTestInvokeStreamServer(t, tc.excType, "please retry")
			defer closeServer()

			role := schemas.ResponsesInputMessageRoleUser
			text := "hello"
			req := &schemas.BifrostResponsesRequest{
				Provider: schemas.Bedrock,
				Model:    "anthropic.claude-sonnet-4-5",
				Input:    []schemas.ResponsesMessage{{Role: &role, Content: &schemas.ResponsesMessageContent{ContentStr: &text}}},
				Params: &schemas.ResponsesParameters{
					ContextManagement: json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`),
				},
			}
			streamChan, bifrostErr := provider.ResponsesStream(testBedrockCtx(), noopPostHookRunner, nil, testBedrockKey(), req)
			require.Nil(t, bifrostErr, "expected the exception to surface as a stream chunk")
			require.NotNil(t, streamChan)

			var errChunk *schemas.BifrostStreamChunk
			for chunk := range streamChan {
				if chunk != nil && chunk.BifrostError != nil {
					errChunk = chunk
					break
				}
			}
			for range streamChan {
			}

			require.NotNil(t, errChunk, "expected error chunk for %s", tc.excType)
			assert.False(t, errChunk.BifrostError.IsBifrostError,
				"%s must be IsBifrostError:false so the retry gate can retry it", tc.excType)
			require.NotNil(t, errChunk.BifrostError.StatusCode,
				"%s must carry a StatusCode for the retry gate", tc.excType)
			assert.Equal(t, tc.expectedStatus, *errChunk.BifrostError.StatusCode,
				"%s must map to HTTP %d", tc.excType, tc.expectedStatus)
		})
	}
}
