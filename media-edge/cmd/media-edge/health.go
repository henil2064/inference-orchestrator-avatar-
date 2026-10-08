package main

import (
	"context"
	"net/http"
)

func healthHandler(shutdownCtx context.Context, ready <-chan struct{}) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-shutdownCtx.Done():
			http.Error(w, "shutting down", http.StatusServiceUnavailable)
			return
		default:
		}
		select {
		case <-ready:
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok\n"))
		default:
			http.Error(w, "starting", http.StatusServiceUnavailable)
		}
	}
}
