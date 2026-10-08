package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"mondee/media-edge/internal/profile"
)

// TestWsHandlerCountsAttemptBeforeAdmissionRejects proves the offered-load
// counter (edge_connection_attempts_total) is incremented before the
// admission decision, and that a rejected attempt is counted separately
// (edge_connection_rejected_total) without ever reaching the WebSocket
// upgrade or edge_sessions_total -- the exact "attempts counter before
// rejection/upgrade" gap the Load & Horizontal-Scaling Plan calls out.
func TestWsHandlerCountsAttemptBeforeAdmissionRejects(t *testing.T) {
	origSlots, origMax := sessionSlots, maxConcurrentSessions
	defer func() { sessionSlots, maxConcurrentSessions = origSlots, origMax }()

	maxConcurrentSessions = 1
	sessionSlots = make(chan struct{}, 1)
	sessionSlots <- struct{}{} // fill the one slot: any new attempt is over capacity

	attemptsBefore := testutil.ToFloat64(mConnectionAttempts)
	rejectedBefore := testutil.ToFloat64(mConnectionRejected)
	sessionsBefore := testutil.ToFloat64(mSessionsTotal)

	handler := wsHandler(context.Background(), profile.Profile{}, nil, nil)
	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodGet, "/ws", nil))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 at capacity, got %d", rec.Code)
	}
	if got := testutil.ToFloat64(mConnectionAttempts) - attemptsBefore; got != 1 {
		t.Errorf("edge_connection_attempts_total: got +%v, want +1", got)
	}
	if got := testutil.ToFloat64(mConnectionRejected) - rejectedBefore; got != 1 {
		t.Errorf("edge_connection_rejected_total: got +%v, want +1", got)
	}
	if got := testutil.ToFloat64(mSessionsTotal) - sessionsBefore; got != 0 {
		t.Errorf("edge_sessions_total must not count a rejected attempt: got +%v, want +0", got)
	}
}

func TestParseMaxSessions(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"", 0},     // unset -> unlimited
		{"0", 0},    // explicit unlimited
		{"5", 5},    // normal
		{"  8 ", 8}, // trimmed
		{"10x", 0},  // trailing garbage -> unlimited, not 10
		{"-3", 0},   // negative -> unlimited
		{"abc", 0},  // non-numeric -> unlimited
		{"2147483647", 2147483647},
	}
	for _, c := range cases {
		if got := parseMaxSessions(c.in); got != c.want {
			t.Errorf("parseMaxSessions(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}
