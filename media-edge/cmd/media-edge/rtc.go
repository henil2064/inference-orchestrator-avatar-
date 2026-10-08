package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/pion/interceptor"
	"github.com/pion/webrtc/v4"
	"google.golang.org/grpc"

	"mondee/media-edge/internal/edgepb"
	"mondee/media-edge/internal/profile"
)

// newPeerConnection builds a receiving peer whose media engine knows about
// Opus and nothing else. The default engine also registers G.722/PCMU/PCMA,
// and codec matching happens on the offer's fmtp line — the instant a
// client's Opus fmtp doesn't match byte-for-byte, negotiation can quietly
// fall back to one of those instead of erroring. Registering only Opus means
// there is nothing to fall back to.
//
// withVideo additionally registers H.264 for the outbound avatar track. It is
// off unless an avatar service is configured, so an edge without one answers
// offers exactly as it always has.
func newPeerConnection(withVideo bool) (*webrtc.PeerConnection, error) {
	m := &webrtc.MediaEngine{}
	if err := m.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeOpus,
			ClockRate:   48000,
			Channels:    2,
			SDPFmtpLine: "minptime=10;useinbandfec=1",
		},
		PayloadType: 111,
	}, webrtc.RTPCodecTypeAudio); err != nil {
		return nil, err
	}
	if withVideo {
		codec := avatarVideoCodec
		// PLI/FIR are how a browser asks for a keyframe after loss; NACK lets
		// pion's interceptors retransmit instead of waiting for one.
		codec.RTCPFeedback = []webrtc.RTCPFeedback{
			{Type: "goog-remb"}, {Type: "ccm", Parameter: "fir"},
			{Type: "nack"}, {Type: "nack", Parameter: "pli"},
		}
		if err := m.RegisterCodec(webrtc.RTPCodecParameters{
			RTPCodecCapability: codec, PayloadType: 102,
		}, webrtc.RTPCodecTypeVideo); err != nil {
			return nil, err
		}
	}

	ir := &interceptor.Registry{}
	if err := webrtc.RegisterDefaultInterceptors(m, ir); err != nil {
		return nil, err
	}

	api := webrtc.NewAPI(webrtc.WithMediaEngine(m), webrtc.WithInterceptorRegistry(ir))
	return api.NewPeerConnection(webrtc.Configuration{
		ICEServers:   []webrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
		BundlePolicy: webrtc.BundlePolicyMaxBundle,
	})
}

// originAllowed checks whether the request origin is permitted. When
// EDGE_ALLOWED_ORIGINS is unset or empty, all origins are accepted (same as
// before — LAN mode). When set, only exact matches are allowed.
func originAllowed(r *http.Request) bool {
	raw := os.Getenv("EDGE_ALLOWED_ORIGINS")
	if raw == "" {
		return true // no restriction — LAN tool
	}
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true // non-browser / same-origin
	}
	for _, allow := range strings.Split(raw, ",") {
		allow = strings.TrimSpace(allow)
		if allow != "" && strings.EqualFold(origin, allow) {
			return true
		}
	}
	return false
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     originAllowed,
}

const (
	wsReadLimit       = 64 << 10
	wsPongTimeout     = 70 * time.Second
	wsPingInterval    = 25 * time.Second
	wsWriteTimeout    = 10 * time.Second
	maxPendingEvents  = 32
	maxPendingBytes   = 64 << 10
	maxAgentEventText = 16 << 10
)

// maxConcurrentSessions caps the number of active voice sessions. When the
// limit is reached, new WebSocket connections receive an error and are closed.
// Set via EDGE_MAX_SESSIONS (0 = unlimited, default for LAN beta).
var maxConcurrentSessions = parseMaxSessions(os.Getenv("EDGE_MAX_SESSIONS"))

// parseMaxSessions reads the EDGE_MAX_SESSIONS cap. Unset, malformed, or
// negative all mean unlimited (0) — the LAN-beta default.
func parseMaxSessions(raw string) int {
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// sessionSlots gates admission: a buffered channel of size N acts as a
// counting semaphore. nil when unlimited.
var sessionSlots = func() chan struct{} {
	if maxConcurrentSessions <= 0 {
		return nil
	}
	return make(chan struct{}, maxConcurrentSessions)
}()

// signalMsg is the wire format exchanged over /ws. offer/answer/candidate
// mirror what RTCSessionDescription/RTCIceCandidate toJSON() produce in the
// browser, so no translation layer is needed.
type signalMsg struct {
	Type      string                     `json:"type"`
	Offer     *webrtc.SessionDescription `json:"offer,omitempty"`
	Answer    *webrtc.SessionDescription `json:"answer,omitempty"`
	Candidate *webrtc.ICECandidateInit   `json:"candidate,omitempty"`

	File      string `json:"file,omitempty"`
	Bytes     int64  `json:"bytes,omitempty"`
	Error     string `json:"error,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	Text      string `json:"text,omitempty"`
	ProcMs    int64  `json:"proc_ms,omitempty"`

	// Agent-turn lifecycle (agent_turn_started / utterance_end /
	// playback_complete). TurnId scopes every signal to one turn so a client
	// cannot mistake a barged-in turn's trailing event for the next turn's.
	TurnID int64  `json:"turn_id,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// wsSession serializes writes to one client's socket — required because both
// the signaling loop and the recording goroutine send on it concurrently.
type wsSession struct {
	conn *websocket.Conn
	mu   sync.Mutex
}

func (s *wsSession) send(v signalMsg) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.conn.SetWriteDeadline(time.Now().Add(wsWriteTimeout))
	if err := s.conn.WriteJSON(v); err != nil {
		errorf("signaling send failed: %v", err)
		return false
	}
	return true
}

func (s *wsSession) ping() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.conn.SetWriteDeadline(time.Now().Add(wsWriteTimeout))
	if err := s.conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(wsWriteTimeout)); err != nil {
		errorf("signaling ping failed: %v", err)
		return false
	}
	return true
}

// rtcEvents is the server side of the browser-created, ordered WebRTC data
// channel. Agent output must travel over the peer connection, not the
// WebSocket that only performs SDP/ICE signaling.
type rtcEvents struct {
	mu           sync.Mutex
	dc           *webrtc.DataChannel
	pending      []string
	pendingBytes int
}

func (e *rtcEvents) setChannel(dc *webrtc.DataChannel) {
	e.mu.Lock()
	e.dc = dc
	e.mu.Unlock()
	dc.OnOpen(func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		for _, message := range e.pending {
			if err := dc.SendText(message); err != nil {
				errorf("agent-events send failed: %v", err)
				break
			}
		}
		e.pending = nil
		e.pendingBytes = 0
	})
}

func (e *rtcEvents) send(v signalMsg) {
	message, err := json.Marshal(v)
	if err != nil {
		errorf("marshal agent event: %v", err)
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.dc == nil || e.dc.ReadyState() != webrtc.DataChannelStateOpen {
		// The browser creates the channel in its offer. Buffer a few early events
		// until SCTP opens instead of dropping a fast backend reply.
		if len(message) <= maxAgentEventText && len(e.pending) < maxPendingEvents && e.pendingBytes+len(message) <= maxPendingBytes {
			e.pending = append(e.pending, string(message))
			e.pendingBytes += len(message)
		} else {
			errorf("agent-events pending buffer full; dropping event type=%s", v.Type)
		}
		return
	}
	if len(message) > maxAgentEventText {
		errorf("agent-events message too large; dropping event type=%s", v.Type)
		return
	}
	if err := e.dc.SendText(string(message)); err != nil {
		errorf("agent-events send failed: %v", err)
	}
}

func (e *rtcEvents) sendUtteranceEnd(sessionID string, turnID int64, reason string) {
	msg := signalMsg{SessionID: sessionID, TurnID: turnID, Reason: reason}
	msg.Type = "utterance_end"
	e.send(msg)
	// Backward compatibility for the existing browser client and any external
	// consumers already listening for this event name.
	msg.Type = "playback_complete"
	e.send(msg)
}

// wsHandler upgrades to a WebSocket and drives one WebRTC session over it:
// trickled offer/answer/ICE, an explicit "stop" from the client to finalize
// the recording immediately (rather than waiting on ICE disconnect timeouts),
// and "saved"/"error" acks so the browser knows the recording actually landed.
//
// avatar is the connection to the optional avatar service; nil disables video.
func wsHandler(shutdownCtx context.Context, prof profile.Profile, orch, avatar *grpc.ClientConn) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Offered load, counted before any admission decision: a generator's
		// schedule is offered demand, not proof it reached the server -- this
		// is the counter that turns it into proof.
		mConnectionAttempts.Inc()

		// Admission control: reject with 503 when at capacity.
		if sessionSlots != nil {
			select {
			case sessionSlots <- struct{}{}:
				defer func() { <-sessionSlots }()
			default:
				mConnectionRejected.Inc()
				http.Error(w, "server busy: at session capacity", http.StatusServiceUnavailable)
				return
			}
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			mUpgradeErrors.Inc()
			errorf("ws upgrade failed: %v", err)
			return
		}
		defer conn.Close()
		conn.SetReadLimit(wsReadLimit)
		conn.SetReadDeadline(time.Now().Add(wsPongTimeout))
		conn.SetPongHandler(func(string) error {
			conn.SetReadDeadline(time.Now().Add(wsPongTimeout))
			return nil
		})

		pc, err := newPeerConnection(avatar != nil)
		if err != nil {
			errorf("new peer connection: %v", err)
			return
		}
		defer pc.Close()

		session := &wsSession{conn: conn}
		agentEvents := &rtcEvents{}

		// Outbound agent-audio plane: one Opus track + playout buffer per
		// session. The track is added to the peer during offer handling below;
		// the playout paces TTS PCM onto it and owns the local barge-in cut.
		// Capability mirrors the registered codec so the sender binds cleanly;
		// we encode mono Opus over it.
		agentTrack, err := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{
			MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2,
			SDPFmtpLine: "minptime=10;useinbandfec=1",
		}, "audio", "agent")
		if err != nil {
			errorf("new agent track: %v", err)
			return
		}
		// The mic-gating signal: fired once the turn's audio has actually left the
		// playout buffer, not when the orchestrator stopped producing it.
		agentPlayout, err := newPlayout(agentTrack, func(sessionID string, turnID int64, reason string) {
			agentEvents.sendUtteranceEnd(sessionID, turnID, reason)
		})
		if err != nil {
			errorf("new playout (is libopus installed?): %v", err)
			return
		}
		defer agentPlayout.Close()
		var agentTrackAdded bool

		// Outbound avatar plane (optional): a video track in the same media
		// stream as the agent audio, so the browser lip-syncs the two. The
		// track exists from the offer; the avatar session that feeds it starts
		// with the audio track, once there is a session id to give it.
		var avatarTrack atomic.Pointer[webrtc.TrackLocalStaticSample]
		var avatarRef atomic.Pointer[avatarSession]

		var activeMu sync.Mutex
		var activeCancel context.CancelFunc
		var activeTrack *webrtc.TrackRemote
		var activeTracks int
		eventf("client connected: %s", r.RemoteAddr)

		pingDone := make(chan struct{})
		defer close(pingDone)
		go func() {
			ticker := time.NewTicker(wsPingInterval)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if !session.ping() {
						return
					}
				case <-pingDone:
					return
				}
			}
		}()

		// On server shutdown, drain this session: cancel the active track so
		// forwardTrack can send SESSION_CLOSE, and close the conn to unblock the
		// read loop below. Without this, srv.Shutdown does not touch hijacked
		// WebSocket conns and the process would hang on sessionWg.Wait().
		go func() {
			select {
			case <-shutdownCtx.Done():
			case <-pingDone:
				return
			}
			activeMu.Lock()
			cancel, track := activeCancel, activeTrack
			activeMu.Unlock()
			if cancel != nil {
				cancel()
			}
			if track != nil {
				_ = track.SetReadDeadline(time.Now())
			}
			_ = conn.Close()
		}()

		pc.OnICECandidate(func(c *webrtc.ICECandidate) {
			if c == nil {
				return
			}
			init := c.ToJSON()
			session.send(signalMsg{Type: "ice", Candidate: &init})
		})

		pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
			eventf("client %s WebRTC state: %s", r.RemoteAddr, s)
			// Force-close on disconnect/failure instead of waiting out ICE's
			// default 5-25s timeouts, so the recording finalizes promptly.
			if s == webrtc.PeerConnectionStateDisconnected || s == webrtc.PeerConnectionStateFailed {
				_ = pc.Close()
			}
		})

		pc.OnDataChannel(func(dc *webrtc.DataChannel) {
			if dc.Label() != "agent-events" {
				return
			}
			agentEvents.setChannel(dc)
		})

		pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
			if track.Kind() != webrtc.RTPCodecTypeAudio {
				return
			}
			activeMu.Lock()
			if activeTracks > 0 {
				activeMu.Unlock()
				session.send(signalMsg{Type: "error", Error: "only one audio track is supported per peer"})
				_ = track.SetReadDeadline(time.Now())
				return
			}
			activeTracks++
			activeMu.Unlock()
			eventf("client %s audio track started", r.RemoteAddr)
			trackCtx, cancelTrack := context.WithCancel(context.Background())
			activeMu.Lock()
			activeCancel = cancelTrack
			activeTrack = track
			activeMu.Unlock()
			defer func() {
				cancelTrack()
				activeMu.Lock()
				if activeTrack == track {
					activeTrack = nil
					activeCancel = nil
				}
				activeTracks--
				activeMu.Unlock()
			}()

			sessionWg.Add(1)
			defer sessionWg.Done()

			var sink playoutSink = agentPlayout
			videoTrack := avatarTrack.Load()
			if videoTrack != nil {
				sink = avatarSink{playout: agentPlayout, session: avatarRef.Load}
				defer func() {
					if a := avatarRef.Swap(nil); a != nil {
						a.Close()
					}
				}()
			}

			sid, n, err := forwardTrack(trackCtx, orch, track, prof, sink,
				func(sessionID string) {
					agentEvents.send(signalMsg{Type: "session_open", SessionID: sessionID})
					if videoTrack != nil {
						if a := startAvatar(avatar, sessionID, videoTrack, agentPlayout); a != nil {
							avatarRef.Store(a)
						}
					}
				},
				func(reply *edgepb.AgentReply) {
					agentEvents.send(signalMsg{Type: "agent_reply", SessionID: reply.SessionId,
						Text: reply.Text, ProcMs: reply.ProcMs, TurnID: reply.TurnId})
				},
				func(sessionID string) { agentEvents.send(signalMsg{Type: "session_complete", SessionID: sessionID}) },
				func(c *edgepb.Control) {
					switch c.Type {
					case edgepb.Control_AGENT_TURN_STARTED:
						agentEvents.send(signalMsg{Type: "agent_turn_started",
							SessionID: c.SessionId, TurnID: c.TurnId})
					case edgepb.Control_PLAYBACK_COMPLETE:
						agentEvents.sendUtteranceEnd(c.SessionId, c.TurnId, c.Metadata["reason"])
					}
				})
			if n == 0 {
				// Nothing forwarded — an "error" ack, not a misleading success.
				errMsg := "no audio forwarded"
				if err != nil {
					errMsg = err.Error()
				}
				session.send(signalMsg{Type: "error", Error: errMsg})
				return
			}
			// Reuse the browser's completion ack: File=session id, Bytes=PCM
			// forwarded to the orchestrator. (No local file is written now.)
			msg := signalMsg{Type: "saved", File: sid, Bytes: n}
			if err != nil {
				msg.Error = err.Error() // partial: streamed some, then errored
				eventf("audio session %s ended with a partial stream: %v", sid, err)
			} else {
				eventf("audio session %s completed (%d bytes forwarded)", sid, n)
			}
			session.send(msg)
		})

		// The read deadline is extended by pong frames from the browser. That
		// gives half-open clients a bounded lifetime even if the WebRTC state
		// callback does not fire promptly.
		for {
			var msg signalMsg
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}
			switch msg.Type {
			case "offer":
				if msg.Offer == nil {
					continue
				}
				if err := pc.SetRemoteDescription(*msg.Offer); err != nil {
					errorf("SetRemoteDescription: %v", err)
					session.send(signalMsg{Type: "error", Error: err.Error()})
					continue
				}
				// Attach the agent-audio track once, after the remote offer is
				// set so pion reuses the browser's sendrecv audio transceiver
				// (the browser's addTrack makes it sendrecv) — the answer then
				// negotiates server->browser audio without a renegotiation.
				if !agentTrackAdded {
					if _, err := pc.AddTrack(agentTrack); err != nil {
						errorf("add agent track: %v", err)
					} else {
						agentTrackAdded = true
					}
				}
				// Same pattern for video, and only when the browser offered to
				// receive it: pion reuses the offer's recvonly transceiver.
				if avatar != nil && avatarTrack.Load() == nil && offerHasVideo(msg.Offer.SDP) {
					videoTrack, err := webrtc.NewTrackLocalStaticSample(avatarVideoCodec, "video", "agent")
					if err != nil {
						errorf("new avatar track: %v", err)
					} else if sender, err := pc.AddTrack(videoTrack); err != nil {
						errorf("add avatar track: %v", err)
					} else {
						avatarTrack.Store(videoTrack)
						go watchKeyframeRequests(sender, avatarRef.Load)
					}
				}
				answer, err := pc.CreateAnswer(nil)
				if err != nil {
					errorf("CreateAnswer: %v", err)
					continue
				}
				// pion rejects SetLocalDescription if the SDP doesn't
				// exactly match what CreateAnswer produced, so the tuned
				// profile params only go into the copy sent over the wire —
				// pion's own view of the answer stays untouched.
				if err := pc.SetLocalDescription(answer); err != nil {
					errorf("SetLocalDescription: %v", err)
					continue
				}
				tuned := *pc.LocalDescription()
				tuned.SDP = profile.TuneSDP(tuned.SDP, prof)
				session.send(signalMsg{Type: "answer", Answer: &tuned})
			case "ice":
				if msg.Candidate == nil {
					continue
				}
				if pc.RemoteDescription() == nil {
					errorf("AddICECandidate before remote description")
					continue
				}
				if err := pc.AddICECandidate(*msg.Candidate); err != nil {
					errorf("AddICECandidate: %v", err)
				}
			case "stop":
				activeMu.Lock()
				cancel := activeCancel
				track := activeTrack
				activeMu.Unlock()
				if cancel != nil {
					cancel()
				}
				if track != nil {
					_ = track.SetReadDeadline(time.Now())
				} else {
					session.send(signalMsg{Type: "error", Error: "stop requested before audio track started"})
				}
			}
		}

		_ = pc.Close()
		eventf("client disconnected: %s", r.RemoteAddr)
	}
}
