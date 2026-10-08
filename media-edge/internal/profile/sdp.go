package profile

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var (
	mLineRe    = regexp.MustCompile(`^m=(\w+)`)
	opusRtpmap = regexp.MustCompile(`^a=rtpmap:(\d+)\s+opus/`)
	fmtpRe     = regexp.MustCompile(`^a=fmtp:(\d+)\s+(.*)$`)
)

// TuneSDP rewrites the audio media section of sdp to apply the profile's
// tested low-latency parameters: ptime, target bitrate, Opus FEC and max
// playback rate. It is a no-op outside the audio m= section and leaves
// video/application sections untouched. Safe to call on both offers and
// answers; only takes effect when p.TuneAnswer is true.
func TuneSDP(sdp string, p Profile) string {
	if !p.TuneAnswer {
		return sdp
	}
	lines := strings.Split(strings.TrimRight(sdp, "\r\n"), "\r\n")
	if len(lines) == 1 {
		lines = strings.Split(sdp, "\n")
	}

	inAudio := false
	opusPT := ""
	out := make([]string, 0, len(lines)+4)

	for i := 0; i < len(lines); i++ {
		line := lines[i]

		if m := mLineRe.FindStringSubmatch(line); m != nil {
			inAudio = m[1] == "audio"
			out = append(out, line)
			if inAudio {
				out = append(out, fmt.Sprintf("b=AS:%d", p.BitrateBps/1000))
			}
			continue
		}

		if !inAudio {
			out = append(out, line)
			continue
		}

		if strings.HasPrefix(line, "b=AS:") {
			continue // replaced above
		}
		if strings.HasPrefix(line, "a=ptime:") {
			continue // replaced below, once, right after fmtp
		}
		if m := opusRtpmap.FindStringSubmatch(line); m != nil {
			opusPT = m[1]
			out = append(out, line)
			continue
		}
		if m := fmtpRe.FindStringSubmatch(line); m != nil && m[1] == opusPT {
			out = append(out, fmt.Sprintf("a=fmtp:%s %s", opusPT, tunedFmtp(m[2], p)))
			out = append(out, "a=ptime:"+strconv.Itoa(p.PtimeMs))
			continue
		}

		out = append(out, line)
	}

	return strings.Join(out, "\r\n") + "\r\n"
}

// tunedFmtp merges FEC/maxplaybackrate/maxaveragebitrate into an existing
// Opus fmtp parameter string, overriding any params it already sets.
func tunedFmtp(fmtp string, p Profile) string {
	params := map[string]string{}
	order := []string{}
	for kv := range strings.SplitSeq(fmtp, ";") {
		kv = strings.TrimSpace(kv)
		if kv == "" {
			continue
		}
		k := kv
		if idx := strings.Index(kv, "="); idx >= 0 {
			k = kv[:idx]
		}
		if _, ok := params[k]; !ok {
			order = append(order, k)
		}
		params[k] = kv
	}

	set := func(k, v string) {
		if _, ok := params[k]; !ok {
			order = append(order, k)
		}
		params[k] = k + "=" + v
	}
	if p.FEC {
		set("useinbandfec", "1")
	} else {
		set("useinbandfec", "0")
	}
	if p.MaxPlaybackHz > 0 {
		set("maxplaybackrate", strconv.Itoa(p.MaxPlaybackHz))
	}
	if p.BitrateBps > 0 {
		set("maxaveragebitrate", strconv.Itoa(p.BitrateBps))
	}

	parts := make([]string, 0, len(order))
	for _, k := range order {
		parts = append(parts, params[k])
	}
	return strings.Join(parts, ";")
}
