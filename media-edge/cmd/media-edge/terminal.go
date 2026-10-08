package main

import (
	"fmt"
	"os"
	"strings"
)

// terminalLevel controls human-facing terminal output. Prometheus metrics are
// unaffected. The default intentionally reports errors only.
var terminalLevel = "ERROR"

func configureTerminal() {
	switch strings.ToUpper(os.Getenv("LOG_LEVEL")) {
	case "DEBUG", "INFO", "WARN", "WARNING", "ERROR":
		terminalLevel = strings.ToUpper(os.Getenv("LOG_LEVEL"))
	}
}

func terminalAllows(level string) bool {
	rank := map[string]int{"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3}
	return rank[level] >= rank[terminalLevel]
}

func terminalf(level, format string, args ...any) {
	if terminalAllows(level) {
		fmt.Fprintf(os.Stderr, "%s media-edge | %s\n", level, fmt.Sprintf(format, args...))
	}
}

func infof(format string, args ...any)  { terminalf("INFO", format, args...) }
func warnf(format string, args ...any)  { terminalf("WARN", format, args...) }
func errorf(format string, args ...any) { terminalf("ERROR", format, args...) }

// eventf is for the small set of connection and call-lifecycle events an
// operator commonly needs during a live test. It is independent of LOG_LEVEL.
func eventf(format string, args ...any) {
	if os.Getenv("EDGE_CONNECTION_EVENTS") == "1" || strings.EqualFold(os.Getenv("EDGE_CONNECTION_EVENTS"), "true") {
		fmt.Fprintf(os.Stdout, "EVENT media-edge | %s\n", fmt.Sprintf(format, args...))
	}
}

func fatalf(format string, args ...any) {
	errorf(format, args...)
	os.Exit(1)
}

// startupBanner is intentionally separate from LOG_LEVEL: a single readiness
// summary is useful in production, while recurring operational chatter is not.
func startupBanner(service string, lines []string) {
	if os.Getenv("STARTUP_BANNER") == "0" {
		return
	}
	width := 62
	fmt.Fprintf(os.Stdout, "\n+%s+\n", strings.Repeat("=", width))
	fmt.Fprintf(os.Stdout, "|  %-60s|\n", service)
	fmt.Fprintf(os.Stdout, "+%s+\n", strings.Repeat("-", width))
	for _, line := range lines {
		fmt.Fprintf(os.Stdout, "|  %-60s|\n", line)
	}
	fmt.Fprintf(os.Stdout, "+%s+\n\n", strings.Repeat("=", width))
}
