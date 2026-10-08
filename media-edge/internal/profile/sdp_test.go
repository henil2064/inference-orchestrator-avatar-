package profile

import (
	"strings"
	"testing"
)

const sampleSDP = "v=0\r\n" +
	"o=- 1 1 IN IP4 127.0.0.1\r\n" +
	"s=-\r\n" +
	"m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n" +
	"c=IN IP4 0.0.0.0\r\n" +
	"a=rtpmap:111 opus/48000/2\r\n" +
	"a=fmtp:111 minptime=10;useinbandfec=0\r\n" +
	"a=ptime:20\r\n" +
	"m=video 9 UDP/TLS/RTP/SAVPF 96\r\n" +
	"a=rtpmap:96 VP8/90000\r\n"

func TestTuneSDP(t *testing.T) {
	out := TuneSDP(sampleSDP, Voice)

	mustContain(t, out, "b=AS:24")
	mustContain(t, out, "a=ptime:10")
	mustContain(t, out, "useinbandfec=1")
	mustContain(t, out, "maxplaybackrate=16000")
	mustContain(t, out, "maxaveragebitrate=24000")
	mustContain(t, out, "minptime=10") // untouched fmtp param preserved
	mustContain(t, out, "a=rtpmap:96 VP8/90000") // video section untouched

	off := Voice
	off.TuneAnswer = false
	if TuneSDP(sampleSDP, off) != sampleSDP {
		t.Fatal("TuneAnswer=false must be a no-op")
	}
}

func mustContain(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Fatalf("expected output to contain %q, got:\n%s", needle, haystack)
	}
}
