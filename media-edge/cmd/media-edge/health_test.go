package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthHandlerReportsStartingReadyAndShutdown(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ready := make(chan struct{})
	handler := healthHandler(ctx, ready)

	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("starting health status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
	}

	close(ready)
	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("ready health status = %d, want %d", rec.Code, http.StatusOK)
	}

	cancel()
	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("shutdown health status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
	}
}
