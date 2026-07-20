// ignore_for_file: avoid_web_libraries_in_flutter
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:math' show min, max;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';
import 'package:web/web.dart' as web;
import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';

// ── Models ───────────────────────────────────────────────────────────────────── //

enum _WsStatus { disconnected, connecting, connected, error }

class _Msg {
  final String text;
  final DateTime time;
  _Msg(this.text) : time = DateTime.now();
}

class _GlossItem {
  final String gloss;
  final double prob;
  const _GlossItem(this.gloss, this.prob);
}

// Same value as services/api_gateway/orchestrator.py's _MIN_CONFIDENCE —
// that constant already gates what enters sentence accumulation server-side;
// this keeps the live gloss chips from showing predictions the server itself
// would discard as noise.
const double _kMinDisplayConfidence = 0.15;

// Two-party call UI hidden for now -- `flutter run -d chrome` opens one
// tab per debug session and a second manually-opened tab against the same
// dev server doesn't get a working Flutter instance (a `flutter run`
// debug-mode limitation, not a bug in the call feature itself), which
// made it impractical to test two participants side by side. Backend
// (call_manager.py, /api/v1/call/*, the WS/REST relay) is untouched and
// still fully working -- this only hides the client entry point so it's
// not reachable from the UI while off. Flip back to true once testing via
// two independent `flutter build web` + static-server tabs (or two real
// devices) instead of two `flutter run` processes.
const bool _kCallFeatureEnabled = false;

// ── HomePage ──────────────────────────────────────────────────────────────────── //

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // ── Camera ───────────────────────────────────────────────────────────────── //
  web.HTMLVideoElement? _video;
  web.HTMLCanvasElement? _canvas;
  web.MediaStream? _mediaStream;
  Timer? _frameTimer;
  bool _cameraActive = false;
  String _cameraError = '';
  // Set instead of _mediaStream when the frame source is an uploaded video
  // file rather than a live webcam (see _startVideoFile) -- same
  // _video/_canvas/_frameTimer/WS pipeline either way, _sendFrame() doesn't
  // care where _video's pixels come from. Used by _stopCamera to release
  // the object URL instead of stopping (nonexistent) MediaStream tracks.
  String? _videoObjectUrl;

  // ── RSL → Text WS ────────────────────────────────────────────────────────── //
  WebSocketChannel? _rslWs;
  StreamSubscription<dynamic>? _rslSub;
  _WsStatus _rslStatus = _WsStatus.disconnected;
  List<_GlossItem> _liveGlosses = [];
  bool _liveGlossesPreview = false; // true = early/provisional, not the final classification
  String _rslAudio = '';
  int? _latencyMs;
  DateTime? _lastFrameSent;
  // Prevents frame queue buildup: only one frame in-flight at a time
  bool _waitingForResponse = false;
  // Server-driven: gesture segmentation is automatic (hand-motion based)
  bool _gestureActive = false;

  // Skeleton overlay — two-buffer animation
  List<List<double>>? _targetKp;  // latest from server
  List<List<double>>? _displayKp; // currently rendered (EMA toward target)
  bool _showSkeleton = false;
  Timer? _animTimer;          // 30fps render tick
  Timer? _hideSkeletonTimer;  // 500ms holdout before hiding skeleton

  // Last fully-recognized gesture's own frames (from cv_service's
  // 'gesture_keypoints', raw/pre-normalizer, same [0,1]-ish coordinates as
  // the live per-frame skeleton) — scrubbable via a slider in
  // _skeletonPanel instead of only ever showing whatever the current live
  // frame happens to be, which stopped being representative within about a
  // second of the gesture ending (especially for an uploaded video that
  // then just sits on its last, often static, frame).
  List<List<List<double>>> _recognizedGestureFrames = [];
  int _recognizedGestureFrameIdx = 0;
  // True while showing a completed gesture's frames instead of the live
  // per-frame feed -- cleared as soon as a new gesture starts (gestureActive
  // flips true again), so live tracking resumes for the next sign.
  bool _reviewingGesture = false;

  // Skeleton playback of typed text->RSL glosses (see _skeletonPanel) —
  // separate from the live-camera path above: these are pre-recorded
  // frames from a real gesture example (services/tts_service/skeleton.py),
  // not noisy live tracking, so they're written straight to _displayKp
  // without the EMA smoothing _animateSkeleton applies to the live feed
  // (smoothing pre-recorded frames would just add lag, not reduce jitter).
  Timer? _textSkeletonTimer;
  List<List<List<double>>> _textSkeletonFrames = [];
  int _textSkeletonIdx = 0;

  // RSL recognition chat (panel 3)
  final List<_Msg> _rslMsgs = [];
  final _rslScrollCtrl = ScrollController();
  String? _lastRslText;
  DateTime? _lastRslAt;

  // Buffered gesture sequence not yet sent to the LLM (server auto-flushes
  // after SENTENCE_PAUSE_SECONDS of silence, but the user can also correct
  // it directly — see _sendWsControl). Each entry is that gesture's top-3
  // candidates; only the top-1 is shown as a chip.
  List<List<_GlossItem>> _pendingPositions = [];

  // ── Text → RSL REST ───────────────────────────────────────────────────────── //
  final _textCtrl = TextEditingController();
  bool _ttsProcessing = false;
  final List<_Msg> _glossMsgs = [];
  // Concatenated sign-clip video for the latest text_to_rsl translation —
  // null if none of the glosses matched a reference clip (see
  // services/tts_service/video.py::SignVideoAssembler).
  String? _signVideoB64;
  // Bumped on every "replay" press -- see _signVideoReplayPanel. The video
  // itself doesn't change, so _signVideoB64 alone as a key wouldn't force
  // _SignVideoPlayer to actually remount/restart; this does.
  int _signVideoReplayNonce = 0;
  final _glossScrollCtrl = ScrollController();

  // ── Mic input (ASR) ──────────────────────────────────────────────────────── //
  web.MediaStream? _micStream;
  web.MediaRecorder? _micRecorder;
  final List<web.Blob> _micChunks = [];
  bool _micRecording = false;
  bool _micProcessing = false;

  final _sessionId = const Uuid().v4();

  // ── Two-party call (MVP — see call_manager.py) ──────────────────────────── //
  // Pairs this session with another via a short code; the other
  // participant's finished translations arrive as 'peer_message' WS events
  // and are appended into _rslMsgs (same bubble list as own recognition
  // results, prefixed to tell them apart) -- no new render widget needed.
  String? _callId;
  final _joinCodeCtrl = TextEditingController();
  bool _callBusy = false;

  // ── Camera ───────────────────────────────────────────────────────────────── //

  Future<void> _startCamera() async {
    // Live camera takes over the skeleton panel -- stop any typed-text
    // playback so the two don't write _displayKp at the same time.
    _textSkeletonTimer?.cancel();
    _textSkeletonTimer = null;
    try {
      // Ideal, not exact -- requests the higher resolution but falls back
      // gracefully to whatever the camera actually supports instead of
      // failing outright. Was `video: true` (browser's default pick, often
      // well below what the webcam supports) drawn into a hardcoded 640x480
      // canvas -- finger keypoints are only a few pixels wide at that size,
      // a likely contributor to the hand-keypoint dropout seen in live
      // sessions (see TRIAGE_MEMORY_GAZETA.md).
      //
      // REVERTED from a 720x1280 portrait attempt (matching the reference
      // clips' ~9:16 training framing) back to plain 1280x720 landscape --
      // real-camera testing showed it made tracking WORSE, not better: on a
      // fixed-sensor landscape webcam the browser can't actually reorient,
      // so _sendFrame's center-crop was cutting a portrait window out of a
      // landscape frame, chopping off hands that moved to either side
      // during signing (RSL routinely uses lateral hand movement) instead
      // of just "showing less of the sides" as assumed. Training-framing
      // match is a real, separate concern, but it doesn't matter if the
      // hands performing the sign are literally outside the captured
      // region. Landscape + full-frame (no crop) keeps the whole arm span
      // visible, which is what real usage confirmed tracking well before.
      // frameRate ideal:30 is a cheap, standards-based nudge against motion
      // blur: most webcams' auto-exposure caps exposure time at ~1/fps, so
      // asking for a higher fps indirectly asks for shorter exposure on
      // fast hand motion -- NOT a guarantee (auto-exposure logic is
      // driver-specific), and deliberately NOT setting exposureTime/
      // exposureMode directly, since manual exposure units are camera-specific
      // and unsupported hardware would need testing before hardcoding a value.
      final stream = await web.window.navigator.mediaDevices
          .getUserMedia(web.MediaStreamConstraints(
            video: web.MediaTrackConstraints(
              width: web.ConstrainULongRange(ideal: 1280),
              height: web.ConstrainULongRange(ideal: 720),
              frameRate: web.ConstrainDoubleRange(ideal: 30),
            ),
            audio: false.toJS,
          ))
          .toDart;
      _mediaStream = stream;
      _video = web.HTMLVideoElement()
        ..srcObject = stream
        ..autoplay = true
        ..muted = true;
      // Off-screen video must be explicitly played in some browsers
      try { await _video!.play().toDart; } catch (_) {}
      // Matches the requested ideal above -- drawImage (in _sendFrame)
      // scales whatever the camera actually delivered into this canvas
      // regardless, so this stays safe even if the camera only supports
      // less than 1280x720.
      _canvas = web.HTMLCanvasElement()..width = 1280..height = 720;
      if (mounted) setState(() => _cameraActive = true);
      await _connectRslWs();
      // Poll frequently (not just every 100ms) so the next frame goes out
      // as soon as the previous response arrives, not on the next fixed
      // tick — _waitingForResponse still guarantees only one frame in
      // flight at a time (server-side session state isn't safe under
      // concurrent frames — see main.py's per-session lock), so this only
      // shrinks dead time between "server is free" and "we notice", it
      // doesn't allow overlapping requests.
      _frameTimer = Timer.periodic(const Duration(milliseconds: 20), (_) => _sendFrame());
      // 30fps animation tick: smoothly interpolate skeleton toward latest keypoints
      _animTimer = Timer.periodic(const Duration(milliseconds: 33), _animateSkeleton);
    } catch (e) {
      if (mounted) setState(() => _cameraError = '$e');
    }
  }

  // Opens a file picker and, on selection, recognizes gestures from the
  // uploaded video instead of a live camera -- same _video/_canvas/WS
  // pipeline as _startCamera below it, _sendFrame() doesn't know or care
  // whether _video's current frame came from a live MediaStream or a
  // <video> playing an uploaded file.
  Future<void> _pickVideoFile() async {
    final input = web.HTMLInputElement()
      ..type = 'file'
      ..accept = 'video/*';
    input.click();
    await input.onChange.first;
    final files = input.files;
    if (files == null || files.length == 0) return;
    final file = files.item(0);
    if (file == null) return;
    await _startVideoFile(file);
  }

  Future<void> _startVideoFile(web.File file) async {
    // Live camera and file playback both drive the same skeleton panel --
    // stop the other one first so they don't fight over _displayKp/_video.
    _textSkeletonTimer?.cancel();
    _textSkeletonTimer = null;
    try {
      final url = web.URL.createObjectURL(file);
      _videoObjectUrl = url;
      _video = web.HTMLVideoElement()
        ..src = url
        ..autoplay = true
        ..muted = true
        ..loop = false;
      try { await _video!.play().toDart; } catch (_) {}
      // Stop automatically once the file finishes playing, same cleanup
      // path as pressing "Стоп" on a live camera.
      _video!.onEnded.listen((_) {
        if (mounted && _cameraActive) _stopCamera();
      });
      _canvas = web.HTMLCanvasElement()..width = 1280..height = 720;
      if (mounted) setState(() => _cameraActive = true);
      await _connectRslWs();
      // Real-time cadence (not "as fast as possible") -- GestureSegmenter's
      // onset/offset thresholds and GESTURE_MIN_FRAMES/MAX_FRAMES are tuned
      // against real hand-motion speed; feeding it frames faster than the
      // video's own real-time playback would look like abnormally fast
      // signing to the segmenter, same concern as the earlier session-wide
      // discussion of segmentation timing.
      _frameTimer = Timer.periodic(const Duration(milliseconds: 20), (_) => _sendFrame());
      _animTimer = Timer.periodic(const Duration(milliseconds: 33), _animateSkeleton);
    } catch (e) {
      if (mounted) setState(() => _cameraError = '$e');
    }
  }

  void _stopCamera() {
    _frameTimer?.cancel();
    _frameTimer = null;
    _animTimer?.cancel();
    _animTimer = null;
    _hideSkeletonTimer?.cancel();
    _hideSkeletonTimer = null;
    _mediaStream?.getTracks().toDart.forEach((t) => t.stop());
    _mediaStream = null;
    if (_videoObjectUrl != null) {
      web.URL.revokeObjectURL(_videoObjectUrl!);
      _videoObjectUrl = null;
    }
    _disconnectRslWs();
    if (mounted) setState(() {
      _cameraActive = false;
      _liveGlosses = [];
      _liveGlossesPreview = false;
      _targetKp = null;
      _displayKp = null;
      _showSkeleton = false;
      _gestureActive = false;
    });
  }

  void _animateSkeleton(Timer _) {
    if (!mounted || _targetKp == null) return;
    const alpha = 0.35; // EMA factor: higher = snappier, lower = smoother
    if (_displayKp == null || _displayKp!.length != _targetKp!.length) {
      // First frame: snap to target immediately
      setState(() => _displayKp = _targetKp!.map((k) => List<double>.from(k)).toList());
      return;
    }
    final next = <List<double>>[];
    for (int i = 0; i < _targetKp!.length; i++) {
      next.add([
        _displayKp![i][0] * (1 - alpha) + _targetKp![i][0] * alpha,
        _displayKp![i][1] * (1 - alpha) + _targetKp![i][1] * alpha,
        _targetKp![i][2], // confidence: use target directly
      ]);
    }
    setState(() => _displayKp = next);
  }

  void _sendFrame() {
    if (_video == null || _canvas == null || _rslStatus != _WsStatus.connected) return;
    // Drop frame if previous one is still being processed (prevents queue buildup → latency spike)
    if (_waitingForResponse) return;
    final ctx = _canvas!.getContext('2d') as web.CanvasRenderingContext2D?;
    if (ctx == null) return;
    // Center-crop (not stretch) the source video to the canvas's aspect
    // ratio before scaling. Canvas target is back to 1280x720 (see
    // _startCamera), same as the requested capture, so in the common case
    // this crops little to nothing -- kept as a safety net for a webcam
    // whose native aspect doesn't match (e.g. 4:3), so it gets a small crop
    // instead of a full-frame stretch, without the earlier portrait-target
    // version's problem of chopping a wide fraction of the frame off.
    final vw = _video!.videoWidth.toDouble();
    final vh = _video!.videoHeight.toDouble();
    final cw = _canvas!.width.toDouble();
    final ch = _canvas!.height.toDouble();
    double sx = 0, sy = 0, sw = vw, sh = vh;
    if (vw > 0 && vh > 0) {
      final srcAspect = vw / vh;
      final dstAspect = cw / ch;
      if (srcAspect > dstAspect) {
        // source wider than target -- crop left/right
        sw = vh * dstAspect;
        sx = (vw - sw) / 2;
      } else if (srcAspect < dstAspect) {
        // source taller than target -- crop top/bottom
        sh = vw / dstAspect;
        sy = (vh - sh) / 2;
      }
    }
    ctx.drawImage(_video!, sx, sy, sw, sh, 0, 0, cw, ch);
    final b64 = _canvas!.toDataURL('image/jpeg', 0.85.toJS).split(',').last;
    _lastFrameSent = DateTime.now();
    _waitingForResponse = true;
    _rslWs!.sink.add(jsonEncode({'type': 'video_frame', 'frame': b64, 'session_id': _sessionId}));
  }

  // ── RSL WS ───────────────────────────────────────────────────────────────── //

  Future<void> _connectRslWs() async {
    setState(() => _rslStatus = _WsStatus.connecting);
    try {
      _rslWs = WebSocketChannel.connect(Uri.parse(Config.wsRslToText));
      await _rslWs!.ready;
      setState(() => _rslStatus = _WsStatus.connected);
      _rslSub = _rslWs!.stream.listen(
        _onRslMessage,
        onError: (_) => setState(() => _rslStatus = _WsStatus.error),
        onDone: () => setState(() => _rslStatus = _WsStatus.disconnected),
      );
    } catch (_) {
      setState(() => _rslStatus = _WsStatus.error);
    }
  }

  void _disconnectRslWs() {
    _rslWs?.sink.add(jsonEncode({'type': 'end_session', 'session_id': _sessionId}));
    _rslSub?.cancel();
    _rslWs?.sink.close();
    _rslWs = null;
    if (mounted) setState(() => _rslStatus = _WsStatus.disconnected);
  }

  void _onRslMessage(dynamic raw) {
    if (!mounted) return;
    final msg = jsonDecode(raw as String) as Map<String, dynamic>;
    final type = msg['type'] as String?;
    final payload = msg['payload'] as Map<String, dynamic>? ?? const {};

    switch (type) {
      case 'gloss':
        final items = (payload['glosses'] as List<dynamic>? ?? [])
            .map((g) => _GlossItem(g['gloss'] as String, (g['prob'] as num).toDouble()))
            .where((g) => g.prob >= _kMinDisplayConfidence)
            .toList();

        // Parse keypoints for skeleton overlay
        final rawKp = payload['keypoints'] as List<dynamic>?;
        List<List<double>>? kp;
        if (rawKp != null) {
          kp = rawKp.map((p) {
            final pts = p as List<dynamic>;
            return [(pts[0] as num).toDouble(), (pts[1] as num).toDouble(), (pts[2] as num).toDouble()];
          }).toList();
        }

        final personDetected = payload['person_detected'] as bool? ?? false;
        final gestureActive = payload['gesture_active'] as bool? ?? false;
        final isPreview = payload['preview'] as bool? ?? false;

        // A new gesture starting means the previous one is no longer
        // "current" -- resume live tracking instead of leaving the last
        // gesture's frames on screen forever.
        if (gestureActive) _reviewingGesture = false;

        // Just-completed gesture's own frames, if this message carries them
        // (cv_service only includes this on the final, non-preview
        // classification) -- freeze the skeleton panel on them instead of
        // the live per-frame feed, scrubbable via the slider in
        // _skeletonPanel.
        final rawGestureKp = payload['gesture_keypoints'] as List<dynamic>?;
        if (rawGestureKp != null && rawGestureKp.isNotEmpty) {
          _recognizedGestureFrames = rawGestureKp.map((frame) {
            return (frame as List<dynamic>).map((p) {
              final pts = p as List<dynamic>;
              return [(pts[0] as num).toDouble(), (pts[1] as num).toDouble(), (pts[2] as num).toDouble()];
            }).toList();
          }).toList();
          _recognizedGestureFrameIdx = 0;
          _reviewingGesture = true;
        }

        // Skeleton holdout: keep showing 500ms after person disappears (avoids flicker)
        if (personDetected) {
          _hideSkeletonTimer?.cancel();
          _hideSkeletonTimer = null;
          _showSkeleton = true;
          if (kp != null && !_reviewingGesture) _targetKp = kp;
        } else if (_showSkeleton && _hideSkeletonTimer == null) {
          _hideSkeletonTimer = Timer(const Duration(milliseconds: 500), () {
            if (mounted) setState(() { _showSkeleton = false; _targetKp = null; _displayKp = null; });
            _hideSkeletonTimer = null;
          });
        }

        setState(() {
          _waitingForResponse = false; // Unblock next frame
          _gestureActive = gestureActive;
          if (_reviewingGesture && _recognizedGestureFrames.isNotEmpty) {
            _displayKp = _recognizedGestureFrames[_recognizedGestureFrameIdx];
          }
          if (items.isNotEmpty) {
            _liveGlosses = items;
            _liveGlossesPreview = isPreview;
          }
          if (_lastFrameSent != null) {
            _latencyMs = DateTime.now().difference(_lastFrameSent!).inMilliseconds;
          }
        });
        break;

      case 'result':
        final text = payload['text'] as String? ?? '';
        if (text.isEmpty) break;
        final now = DateTime.now();
        final isDup = text == _lastRslText &&
            _lastRslAt != null &&
            now.difference(_lastRslAt!) < const Duration(seconds: 3);
        if (!isDup) {
          _lastRslText = text;
          _lastRslAt = now;
          setState(() => _rslMsgs.add(_Msg(text)));
          _scrollToBottom(_rslScrollCtrl);
        }
        break;

      case 'pending_sentence':
        final rawPositions = payload['positions'] as List<dynamic>? ?? [];
        setState(() {
          _pendingPositions = rawPositions.map((pos) {
            return (pos as List<dynamic>)
                .map((g) => _GlossItem(g['gloss'] as String, (g['prob'] as num).toDouble()))
                .toList();
          }).toList();
        });
        break;

      case 'audio':
        setState(() => _rslAudio = payload['wav'] as String? ?? '');
        _playAudio(_rslAudio);
        break;

      case 'video':
        final videoB64 = payload['video'] as String?;
        setState(() => _signVideoB64 = (videoB64 != null && videoB64.isNotEmpty) ? videoB64 : null);
        break;

      case 'skeleton':
        final sequences = payload['sequences'] as List<dynamic>?;
        if (sequences != null && sequences.isNotEmpty) _playTextSkeleton(sequences);
        break;

      case 'error':
        setState(() => _rslStatus = _WsStatus.error);
        break;

      case 'peer_message':
        final text = payload['text'] as String? ?? '';
        final audio = payload['audio'] as String?;
        final video = payload['video'] as String?;
        final peerSkeletons = payload['skeleton_sequences'] as List<dynamic>?;
        final hasSkeletons = peerSkeletons != null && peerSkeletons.isNotEmpty;
        if (text.isEmpty && (audio == null || audio.isEmpty) &&
            (video == null || video.isEmpty) && !hasSkeletons) break;
        setState(() {
          _rslMsgs.add(_Msg(text.isNotEmpty ? '👥 $text' : '👥 (аудио/видео)'));
          if (video != null && video.isNotEmpty) _signVideoB64 = video;
        });
        _scrollToBottom(_rslScrollCtrl);
        if (audio != null && audio.isNotEmpty) _playAudio(audio);
        if (hasSkeletons) _playTextSkeleton(peerSkeletons);
        break;
    }
  }

  // ── Two-party call (MVP) ─────────────────────────────────────────────────── //

  Future<void> _createCall() async {
    setState(() => _callBusy = true);
    try {
      final resp = await http.post(
        Uri.parse('${Config.httpBase}/api/v1/call/create'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': _sessionId}),
      );
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      if (mounted && data['call_id'] != null) {
        setState(() => _callId = data['call_id'] as String);
      }
    } catch (_) {
    } finally {
      if (mounted) setState(() => _callBusy = false);
    }
  }

  Future<void> _joinCall() async {
    final code = _joinCodeCtrl.text.trim().toUpperCase();
    if (code.isEmpty) return;
    setState(() => _callBusy = true);
    try {
      final resp = await http.post(
        Uri.parse('${Config.httpBase}/api/v1/call/$code/join'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'session_id': _sessionId}),
      );
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body) as Map<String, dynamic>;
        if (mounted) setState(() => _callId = data['call_id'] as String);
      }
    } catch (_) {
    } finally {
      if (mounted) setState(() => _callBusy = false);
    }
  }

  void _showCallDialog(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (dialogCtx) => AlertDialog(
        title: const Text('Двусторонний звонок'),
        content: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Text('Один участник создаёт звонок и диктует код второму.',
              style: TextStyle(fontSize: 12)),
          const SizedBox(height: 12),
          FilledButton.icon(
            icon: const Icon(Icons.add_call, size: 18),
            label: const Text('Создать звонок'),
            onPressed: _callBusy ? null : () async {
              await _createCall();
              if (dialogCtx.mounted) Navigator.of(dialogCtx).pop();
            },
          ),
          const SizedBox(height: 16),
          const Text('— или —', style: TextStyle(fontSize: 11)),
          const SizedBox(height: 8),
          TextField(
            controller: _joinCodeCtrl,
            textCapitalization: TextCapitalization.characters,
            decoration: const InputDecoration(
              labelText: 'Код звонка', border: OutlineInputBorder(), isDense: true,
            ),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            icon: const Icon(Icons.login, size: 18),
            label: const Text('Присоединиться'),
            onPressed: _callBusy ? null : () async {
              await _joinCall();
              if (dialogCtx.mounted) Navigator.of(dialogCtx).pop();
            },
          ),
        ]),
        actions: [
          TextButton(onPressed: () => Navigator.of(dialogCtx).pop(), child: const Text('Закрыть')),
        ],
      ),
    );
  }

  // ── Text → RSL REST ──────────────────────────────────────────────────────── //

  // `override` lets mic input (_finishMicRecording) push transcribed text
  // through the exact same translate call as typed text, instead of
  // duplicating this method's body for a second input source.
  Future<void> _sendText({String? override}) async {
    final text = override ?? _textCtrl.text.trim();
    if (text.isEmpty) return;
    if (override == null) _textCtrl.clear();
    setState(() => _ttsProcessing = true);
    try {
      final resp = await http
          .post(
            Uri.parse('${Config.httpBase}/api/v1/translate'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'mode': 'text_to_rsl', 'text': text, 'session_id': _sessionId}),
          )
          .timeout(const Duration(seconds: 30));
      if (resp.statusCode != 200) {
        throw Exception('Сервер ответил ${resp.statusCode}');
      }
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      final glosses = data['translation'] as String? ?? '';
      final videoB64 = data['video_mp4'] as String?;
      final skeletonSeqs = data['skeleton_sequences'] as List<dynamic>?;
      if (glosses.isNotEmpty && mounted) {
        setState(() {
          _glossMsgs.add(_Msg(glosses));
          _signVideoB64 = (videoB64 != null && videoB64.isNotEmpty) ? videoB64 : null;
        });
        _scrollToBottom(_glossScrollCtrl);
      } else if (mounted) {
        // Empty translation isn't a transport error (backend responded
        // fine, NLP just returned nothing useful) -- still worth telling
        // the user instead of leaving the panel looking like the button
        // silently did nothing.
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Пустой ответ от сервера — перевод не получен')),
        );
      }
      if (skeletonSeqs != null && skeletonSeqs.isNotEmpty) _playTextSkeleton(skeletonSeqs);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Не удалось отправить: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _ttsProcessing = false);
    }
  }

  // ── Mic input (ASR) ──────────────────────────────────────────────────────── //
  //
  // Record -> POST /api/v1/asr (speech to text only) -> feed the transcript
  // through _sendText() exactly as if it had been typed, so the rest of the
  // text_to_rsl pipeline (NLP reverse, TTS audio+video+skeleton) isn't
  // duplicated for this input source.
  Future<void> _toggleMic() async {
    if (_micRecording) {
      _micRecorder?.stop();
      return; // rest happens in the 'stop' listener set up below
    }
    try {
      final stream = await web.window.navigator.mediaDevices
          .getUserMedia(web.MediaStreamConstraints(audio: true.toJS))
          .toDart;
      _micStream = stream;
      _micChunks.clear();
      final recorder = web.MediaRecorder(stream);
      const web.EventStreamProvider<web.BlobEvent>('dataavailable')
          .forTarget(recorder)
          .listen((e) => _micChunks.add(e.data));
      const web.EventStreamProvider<web.Event>('stop')
          .forTarget(recorder)
          .listen((_) => _finishMicRecording());
      recorder.start();
      _micRecorder = recorder;
      if (mounted) setState(() => _micRecording = true);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Микрофон недоступен: $e')),
        );
      }
    }
  }

  Future<void> _finishMicRecording() async {
    _micStream?.getTracks().toDart.forEach((t) => t.stop());
    _micStream = null;
    _micRecorder = null;
    if (mounted) setState(() { _micRecording = false; _micProcessing = true; });

    try {
      if (_micChunks.isEmpty) return;
      final blob = _micChunks.length == 1
          ? _micChunks.first
          : web.Blob(_micChunks.map((b) => b as web.BlobPart).toList().toJS);
      final buffer = await blob.arrayBuffer().toDart;
      final bytes = buffer.toDart.asUint8List();
      final audioB64 = base64Encode(bytes);

      final resp = await http
          .post(
            Uri.parse('${Config.httpBase}/api/v1/asr'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'audio': audioB64, 'session_id': _sessionId}),
          )
          .timeout(const Duration(seconds: 30));
      if (resp.statusCode != 200) {
        throw Exception('Сервер ответил ${resp.statusCode}');
      }
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      final text = (data['text'] as String? ?? '').trim();
      if (text.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Речь не распознана')),
          );
        }
        return;
      }
      await _sendText(override: text);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Не удалось распознать речь: $e')),
        );
      }
    } finally {
      _micChunks.clear();
      if (mounted) setState(() => _micProcessing = false);
    }
  }

  // ── Skeleton playback of typed text->RSL glosses ─────────────────────────── //
  //
  // `sequences` is [{"gloss": str, "frames": [[[x,y,conf],...75],...T]}, ...]
  // from tts_service's /skeleton_sequence (see orchestrator.py's
  // _render_skeleton_sequence). Plays each gloss's frames back to back at a
  // fixed rate into the same _skeletonPanel that shows the live camera feed
  // -- whichever source wrote _displayKp most recently is what's on screen;
  // starting the camera cancels this timer (see _startCamera) so the two
  // don't fight over the same panel.
  void _playTextSkeleton(List<dynamic> sequences) {
    _textSkeletonTimer?.cancel();
    final frames = <List<List<double>>>[];
    for (final seq in sequences) {
      final raw = (seq as Map<String, dynamic>)['frames'] as List<dynamic>? ?? [];
      for (final frame in raw) {
        frames.add((frame as List<dynamic>)
            .map((p) => (p as List<dynamic>).map((v) => (v as num).toDouble()).toList())
            .toList());
      }
      // Short pause (zero-confidence frame, renders as "nobody") between
      // glosses so consecutive signs don't blend into one motion blur.
      if (frames.isNotEmpty) {
        frames.add(List<List<double>>.generate(75, (_) => [0.0, 0.0, 0.0]));
      }
    }
    if (frames.isEmpty) return;

    _textSkeletonFrames = frames;
    _textSkeletonIdx = 0;
    setState(() {
      _showSkeleton = true;
      _displayKp = _textSkeletonFrames[0];
    });
    // ~25fps -- matches the WINDOW_SIZE=64-frame clips' original capture
    // cadence closely enough for a natural-looking playback speed.
    _textSkeletonTimer = Timer.periodic(const Duration(milliseconds: 40), (timer) {
      _textSkeletonIdx++;
      if (_textSkeletonIdx >= _textSkeletonFrames.length) {
        timer.cancel();
        return;
      }
      if (!mounted) { timer.cancel(); return; }
      setState(() => _displayKp = _textSkeletonFrames[_textSkeletonIdx]);
    });
  }

  void _playAudio(String base64Wav) {
    if (base64Wav.isEmpty) return;
    (web.HTMLAudioElement()..src = 'data:audio/wav;base64,$base64Wav').play();
  }

  // Manual sentence-buffer controls — bypass the SENTENCE_PAUSE_SECONDS
  // auto-flush fallback (see orchestrator.py::flush_pending_sentence /
  // delete_last_pending). No-ops if the RSL WS isn't connected.
  void _sendWsControl(String type) {
    if (_rslStatus != _WsStatus.connected || _rslWs == null) return;
    _rslWs!.sink.add(jsonEncode({'type': type, 'session_id': _sessionId}));
  }

  void _flushPendingSentence() => _sendWsControl('flush_sentence');
  void _deleteLastPendingGesture() => _sendWsControl('delete_last_gesture');

  void _scrollToBottom(ScrollController ctrl) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (ctrl.hasClients) {
        ctrl.animateTo(ctrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
      }
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isWide = MediaQuery.of(context).size.width >= 800;

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 12,
        toolbarHeight: 48,
        title: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.sign_language, color: cs.primary, size: 20),
          const SizedBox(width: 6),
          const Text('Glossa', style: TextStyle(fontSize: 16)),
          const SizedBox(width: 8),
          _WsDot(status: _rslStatus),
          if (_latencyMs != null) ...[
            const SizedBox(width: 6),
            _LatencyBadge(ms: _latencyMs!),
          ],
        ]),
        actions: [
          if (_kCallFeatureEnabled)
            _callId != null
                ? Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    child: Chip(
                      avatar: const Icon(Icons.call, size: 16),
                      label: Text('Звонок: $_callId', style: const TextStyle(fontSize: 12)),
                      visualDensity: VisualDensity.compact,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                  )
                : IconButton(
                    tooltip: 'Двусторонний звонок',
                    icon: const Icon(Icons.call, size: 20),
                    onPressed: _callBusy ? null : () => _showCallDialog(context),
                  ),
          SizedBox(
            width: 210,
            child: _UrlField(
              initialUrl: Config.serverUrl,
              onChanged: (url) => Config.serverUrl = url,
            ),
          ),
          const SizedBox(width: 10),
        ],
      ),
      // 2x2 quadrants on wide screens: top-left camera (person framed --
      // head/hands/chest, via the framing guide), bottom-left the skeleton
      // rendered on its own (not overlaid), top-right recognized
      // gestures + the LLM-produced sentence chat, bottom-right the
      // outgoing text->RSL panel (typed message -> gloss/video, existing
      // _glossChatPanel/_textInputPanel -- the remaining major feature not
      // otherwise placed by the 3 quadrants that were specified).
      body: isWide
          ? Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  Expanded(child: _cameraPanel(cs)),
                  Divider(height: 1, color: cs.outlineVariant),
                  Expanded(child: _visualOutputPanel(cs)),
                ]),
              ),
              VerticalDivider(width: 1, color: cs.outlineVariant),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                  Expanded(child: _recognitionChatPanel(cs)),
                  Divider(height: 1, color: cs.outlineVariant),
                  Expanded(child: _glossChatPanel(cs)),
                  _textInputPanel(cs),
                ]),
              ),
            ])
          : Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              SizedBox(height: 220, child: _cameraPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              SizedBox(height: 180, child: _visualOutputPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              SizedBox(height: 240, child: _recognitionChatPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              SizedBox(height: 240, child: _glossChatPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              _textInputPanel(cs),
            ]),
    );
  }

  // ── Panel 1: Camera with skeleton overlay ────────────────────────────────── //

  Widget _cameraPanel(ColorScheme cs) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Video feed
        ClipRect(
          child: _cameraActive && _video != null
              ? _CameraView(videoElement: _video!)
              : ColoredBox(
                  color: cs.surfaceContainerHighest,
                  child: Center(
                    child: _cameraError.isNotEmpty
                        ? Padding(
                            padding: const EdgeInsets.all(12),
                            child: Text(_cameraError,
                                style: TextStyle(color: cs.error), textAlign: TextAlign.center),
                          )
                        : Column(mainAxisSize: MainAxisSize.min, children: [
                            Icon(Icons.videocam_off_outlined, size: 36, color: cs.outline),
                            const SizedBox(height: 6),
                            Text('Камера', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
                          ]),
                  ),
                ),
        ),

        // ── Framing guide — "stand here" hint: head/hands/chest inside the
        // box, shown until a person is confidently tracked. The model was
        // trained on Slovo clips framed roughly waist-up, centered, frontal
        // — matching that framing reduces the live/train distribution gap
        // more than any inference post-processing can. Fades out once real
        // tracking kicks in. The skeleton itself now renders in its own
        // panel (see _skeletonPanel) instead of overlaid here, so this
        // panel stays a clean, undistorted view of the actual camera feed. ── //
        if (_cameraActive && !_showSkeleton)
          const CustomPaint(painter: _FramingGuidePainter()),

        // ── Live gloss chips (top of frame) ──────────────────────────────── //
        if (_liveGlosses.isNotEmpty)
          Positioned(
            top: 6, left: 6, right: 6,
            child: Wrap(
              spacing: 4, runSpacing: 4,
              children: _liveGlosses.asMap().entries.map((e) {
                final isTop = e.key == 0;
                final pct = (e.value.prob * 100).toStringAsFixed(0);
                // Preview chips (early, provisional — before the gesture has
                // fully ended) render dimmed with a dashed border and a
                // trailing "…" so it's visually obvious this may still change.
                final baseColor = isTop ? cs.primary : Colors.black54;
                final alpha = _liveGlossesPreview ? 0.5 : (isTop ? 0.88 : 1.0);
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: baseColor.withValues(alpha: alpha),
                    borderRadius: BorderRadius.circular(12),
                    border: _liveGlossesPreview
                        ? Border.all(color: Colors.white.withValues(alpha: 0.6), width: 1)
                        : null,
                  ),
                  child: Text(
                      _liveGlossesPreview ? '${e.value.gloss} $pct%…' : '${e.value.gloss} $pct%',
                      style: TextStyle(
                        color: isTop ? cs.onPrimary : Colors.white,
                        fontSize: 12,
                        fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
                        fontStyle: _liveGlossesPreview ? FontStyle.italic : FontStyle.normal,
                      )),
                );
              }).toList(),
            ),
          ),

        // ── Camera + gesture buttons (bottom) ────────────────────────────── //
        Positioned(
          bottom: 8, left: 0, right: 0,
          child: Center(
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              // Toggle camera
              FilledButton.icon(
                onPressed: _cameraActive ? _stopCamera : _startCamera,
                icon: Icon(_cameraActive ? Icons.stop : Icons.videocam, size: 16),
                label: Text(_cameraActive ? 'Стоп' : 'Камера',
                    style: const TextStyle(fontSize: 13)),
                style: FilledButton.styleFrom(
                  backgroundColor: _cameraActive
                      ? cs.error.withValues(alpha: 0.9)
                      : cs.primary.withValues(alpha: 0.9),
                  foregroundColor: _cameraActive ? cs.onError : cs.onPrimary,
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
              ),
              if (!_cameraActive) ...[
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: _pickVideoFile,
                  icon: const Icon(Icons.upload_file, size: 16),
                  label: const Text('Видео', style: TextStyle(fontSize: 13)),
                  style: OutlinedButton.styleFrom(
                    backgroundColor: Colors.white.withValues(alpha: 0.85),
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ),
              ],
              if (_cameraActive) ...[
                const SizedBox(width: 8),
                // Passive status pill — gesture detection is automatic
                // (server-side hand-motion segmentation), no touch needed.
                AnimatedContainer(
                  duration: const Duration(milliseconds: 150),
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: _gestureActive
                        ? Colors.redAccent.withValues(alpha: 0.92)
                        : Colors.white.withValues(alpha: 0.85),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: _gestureActive ? Colors.redAccent : cs.outline,
                      width: _gestureActive ? 2 : 1,
                    ),
                  ),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    Icon(
                      _gestureActive ? Icons.fiber_manual_record : Icons.pan_tool_outlined,
                      size: 14,
                      color: _gestureActive ? Colors.white : cs.onSurface,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      _gestureActive ? 'Распознаю жест…' : 'Готово к жесту',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: _gestureActive ? Colors.white : cs.onSurface,
                      ),
                    ),
                  ]),
                ),
              ],
            ]),
          ),
        ),
      ],
    );
  }

  // ── Panel 1b: Visual output (skeleton / reference clip / upload preview) ── //
  //
  // One panel, three mutually-exclusive modes depending on what's actually
  // happening right now:
  //  - Live camera signing -> skeleton (own quadrant, not overlaid on the
  //    camera feed -- see _skeletonPanel).
  //  - An uploaded video is being recognized -> no skeleton at all (the
  //    video itself already plays in the top-left camera panel via the
  //    same _CameraView/_video; recognized glosses go to the normal
  //    "Распознанные жесты" chat exactly like live camera does) -- this
  //    panel just shows a neutral "processing" placeholder instead of a
  //    skeleton that used to flash for about a second and mean nothing.
  //  - Neither camera nor upload is active, but the last text->RSL
  //    translation produced a reference-clip video -> show that clip full
  //    size instead of the cramped 180px strip it used to sit in inside
  //    the gloss chat panel, with an explicit replay button.
  bool get _isVideoFileSession => _videoObjectUrl != null;

  Widget _visualOutputPanel(ColorScheme cs) {
    if (_cameraActive && _isVideoFileSession) return _uploadPreviewPanel(cs);
    if (_cameraActive) return _skeletonPanel(cs);
    if (_signVideoB64 != null) return _signVideoReplayPanel(cs);
    return _skeletonPanel(cs); // idle placeholder state (no camera, no clip yet)
  }

  Widget _uploadPreviewPanel(ColorScheme cs) => const ColoredBox(
    color: Colors.black,
    child: Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        SizedBox(
          width: 28, height: 28,
          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white70),
        ),
        SizedBox(height: 10),
        Text('Распознаю жесты в видео…',
            style: TextStyle(color: Colors.white70, fontSize: 12)),
        SizedBox(height: 4),
        Text('Видео — сверху, результат — в чате справа',
            style: TextStyle(color: Colors.white38, fontSize: 11, fontStyle: FontStyle.italic)),
      ]),
    ),
  );

  Widget _signVideoReplayPanel(ColorScheme cs) => ColoredBox(
    color: Colors.black,
    child: Stack(
      fit: StackFit.expand,
      children: [
        // Key includes the replay nonce -- the video itself doesn't change
        // on replay, so without a changing key Flutter would keep the same
        // element mounted instead of actually restarting playback.
        _SignVideoPlayer(
          key: ValueKey('$_signVideoB64-$_signVideoReplayNonce'),
          base64Mp4: _signVideoB64!,
        ),
        const Positioned(
          top: 6, left: 8,
          child: Text('Жесты собеседнику', style: TextStyle(color: Colors.white54, fontSize: 11)),
        ),
        Positioned(
          bottom: 8, right: 8,
          child: FloatingActionButton.small(
            heroTag: 'replaySignVideo',
            tooltip: 'Проиграть заново',
            onPressed: () => setState(() => _signVideoReplayNonce++),
            child: const Icon(Icons.replay),
          ),
        ),
      ],
    ),
  );

  Widget _skeletonPanel(ColorScheme cs) {
    final scrubbable = _reviewingGesture && _recognizedGestureFrames.length > 1;
    return ColoredBox(
      color: Colors.black,
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (_showSkeleton && _displayKp != null)
            CustomPaint(
              painter: _SkeletonPainter(
                keypoints: _displayKp!,
                personDetected: _showSkeleton,
              ),
            )
          else
            Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.accessibility_new, size: 36, color: cs.outline),
                const SizedBox(height: 6),
                Text(
                  _cameraActive ? 'Ищу человека в кадре…' : 'Скелет',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ]),
            ),
          Positioned(
            top: 6, left: 8,
            child: Text(
              _reviewingGesture ? 'Распознанный жест' : 'Точки жеста',
              style: const TextStyle(color: Colors.white54, fontSize: 11),
            ),
          ),
          // Frame-by-frame scrubber for the last recognized gesture --
          // "жест состоит из нескольких кадров" (RSL signs are motion, not
          // a single pose; letting someone step through the frames makes it
          // possible to actually see what was recognized instead of just
          // one static instant of it).
          if (scrubbable)
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: Container(
                color: Colors.black54,
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                child: Row(children: [
                  Text(
                    '${_recognizedGestureFrameIdx + 1}/${_recognizedGestureFrames.length}',
                    style: const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Expanded(
                    child: Slider(
                      value: _recognizedGestureFrameIdx.toDouble(),
                      min: 0,
                      max: (_recognizedGestureFrames.length - 1).toDouble(),
                      divisions: _recognizedGestureFrames.length - 1,
                      onChanged: (v) => setState(() {
                        _recognizedGestureFrameIdx = v.round();
                        _displayKp = _recognizedGestureFrames[_recognizedGestureFrameIdx];
                      }),
                    ),
                  ),
                ]),
              ),
            ),
        ],
      ),
    );
  }

  // ── Panel 2: Gloss chat ───────────────────────────────────────────────────── //

  Widget _glossChatPanel(ColorScheme cs) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      _PanelHeader(icon: Icons.sign_language, label: 'Глоссы от собеседника',
          color: cs.secondary, bg: cs.secondaryContainer.withValues(alpha: 0.5)),
      // Video itself now lives in the full-size visual-output panel
      // (bottom-left, see _visualOutputPanel/_signVideoReplayPanel) --
      // squeezed into a 180px strip here made it look tiny for no reason,
      // since that panel already has a whole quadrant of room to work with.
      Expanded(
        child: _glossMsgs.isEmpty
            ? _EmptyState(icon: Icons.chat_bubble_outline,
                label: 'Глоссы появятся когда\nсобеседник напишет сообщение', cs: cs)
            : ListView.builder(
                controller: _glossScrollCtrl,
                padding: const EdgeInsets.fromLTRB(10, 6, 10, 10),
                itemCount: _glossMsgs.length,
                itemBuilder: (_, i) => _GlossBubble(msg: _glossMsgs[i], cs: cs),
              ),
      ),
    ],
  );

  // ── Panel 3: Recognition chat ─────────────────────────────────────────────── //

  Widget _recognitionChatPanel(ColorScheme cs) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      _PanelHeader(icon: Icons.hearing, label: 'Распознанные жесты',
          color: cs.tertiary, bg: cs.tertiaryContainer.withValues(alpha: 0.5)),
      // Накопленное, ещё не отправленное в LLM предложение. Сервер сам
      // отправит его после паузы (SENTENCE_PAUSE_SECONDS), но можно
      // поправить/отправить раньше вручную — на случай если распознавание
      // ошиблось или пользователь просто не хочет ждать паузы.
      if (_pendingPositions.isNotEmpty)
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
          color: cs.tertiaryContainer.withValues(alpha: 0.25),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Wrap(
              spacing: 4, runSpacing: 4,
              children: _pendingPositions.asMap().entries.map((e) {
                final isLast = e.key == _pendingPositions.length - 1;
                final top1 = e.value.first;
                return Chip(
                  label: Text(top1.gloss, style: const TextStyle(fontSize: 12)),
                  backgroundColor: cs.tertiaryContainer,
                  visualDensity: VisualDensity.compact,
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  onDeleted: isLast ? _deleteLastPendingGesture : null,
                );
              }).toList(),
            ),
            const SizedBox(height: 6),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.tonalIcon(
                onPressed: _flushPendingSentence,
                icon: const Icon(Icons.send, size: 14),
                label: const Text('Отправить сейчас', style: TextStyle(fontSize: 12)),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
              ),
            ),
          ]),
        ),
      Expanded(
        child: _rslMsgs.isEmpty
            ? _EmptyState(icon: Icons.sign_language,
                label: 'Здесь появится перевод жестов\nпосле запуска камеры', cs: cs)
            : ListView.builder(
                controller: _rslScrollCtrl,
                padding: const EdgeInsets.fromLTRB(10, 6, 10, 10),
                itemCount: _rslMsgs.length,
                itemBuilder: (_, i) => _RecognitionBubble(
                  msg: _rslMsgs[i], cs: cs,
                  onPlay: _rslAudio.isNotEmpty && i == _rslMsgs.length - 1
                      ? () => _playAudio(_rslAudio) : null,
                ),
              ),
      ),
    ],
  );

  // ── Panel 4: Text input ───────────────────────────────────────────────────── //

  Widget _textInputPanel(ColorScheme cs) => Container(
    constraints: const BoxConstraints(minHeight: 130, maxHeight: 180),
    color: cs.surface,
    padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
    child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Expanded(
        child: TextField(
          controller: _textCtrl,
          maxLines: null, expands: true,
          textAlignVertical: TextAlignVertical.top,
          decoration: InputDecoration(
            isDense: true,
            contentPadding: const EdgeInsets.all(10),
            border: const OutlineInputBorder(),
            hintText: 'Написать собеседнику…',
            hintStyle: TextStyle(color: cs.onSurfaceVariant, fontSize: 13),
          ),
          style: const TextStyle(fontSize: 14),
          onSubmitted: (_) => _sendText(),
        ),
      ),
      const SizedBox(height: 6),
      Row(children: [
        SizedBox(
          height: 36,
          child: FilledButton.tonalIcon(
            onPressed: _micProcessing ? null : _toggleMic,
            icon: _micProcessing
                ? const SizedBox(width: 14, height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : Icon(_micRecording ? Icons.stop : Icons.mic, size: 16),
            label: Text(_micRecording ? 'Стоп' : 'Микрофон', style: const TextStyle(fontSize: 13)),
            style: _micRecording
                ? FilledButton.styleFrom(
                    backgroundColor: cs.error.withValues(alpha: 0.9), foregroundColor: cs.onError)
                : null,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: SizedBox(
            height: 36,
            child: FilledButton.icon(
              onPressed: _ttsProcessing ? null : _sendText,
              icon: _ttsProcessing
                  ? const SizedBox(width: 14, height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.send, size: 16),
              label: const Text('Отправить', style: TextStyle(fontSize: 13)),
            ),
          ),
        ),
      ]),
    ]),
  );

  @override
  void dispose() {
    _stopCamera();
    _animTimer?.cancel();
    _hideSkeletonTimer?.cancel();
    _textSkeletonTimer?.cancel();
    _micStream?.getTracks().toDart.forEach((t) => t.stop());
    _rslSub?.cancel();
    _rslWs?.sink.close();
    _textCtrl.dispose();
    _joinCodeCtrl.dispose();
    _rslScrollCtrl.dispose();
    _glossScrollCtrl.dispose();
    super.dispose();
  }
}

// ── Framing guide painter ─────────────────────────────────────────────────────── //

/// Dashed "stand here" guide: a waist-up, centered box roughly matching how
/// the Slovo training clips are framed, plus a head-position dot. Purely a
/// visual hint for the user — no effect on the pipeline itself.
class _FramingGuidePainter extends CustomPainter {
  const _FramingGuidePainter();

  void _drawDashedRRect(Canvas canvas, RRect rrect, Paint paint) {
    final path = Path()..addRRect(rrect);
    const dashLen = 8.0, gapLen = 6.0;
    for (final metric in path.computeMetrics()) {
      var distance = 0.0;
      while (distance < metric.length) {
        final next = (distance + dashLen).clamp(0.0, metric.length);
        canvas.drawPath(metric.extractPath(distance, next), paint);
        distance = next + gapLen;
      }
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    // Waist-up framing box: centered, ~55% width, ~80% height, top-aligned
    // with a little headroom — approximates typical Slovo-clip framing.
    final boxW = size.width * 0.55;
    final boxH = size.height * 0.8;
    final rect = Rect.fromCenter(
      center: Offset(size.width / 2, size.height * 0.52),
      width: boxW,
      height: boxH,
    );
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(16));

    _drawDashedRRect(
      canvas,
      rrect,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.55)
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke,
    );

    // Head hint circle near the top of the box
    final headCenter = Offset(size.width / 2, rect.top + boxH * 0.12);
    canvas.drawCircle(
      headCenter,
      boxH * 0.07,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.4)
        ..strokeWidth = 1.5
        ..style = PaintingStyle.stroke,
    );
  }

  @override
  bool shouldRepaint(_FramingGuidePainter old) => false;
}

// ── Skeleton painter ──────────────────────────────────────────────────────────── //

class _SkeletonPainter extends CustomPainter {
  final List<List<double>> keypoints; // (75, 3) — [x, y, score] normalized 0-1
  final bool personDetected;

  const _SkeletonPainter({required this.keypoints, required this.personDetected});

  // COCO-17 body skeleton connections (indices 0-16)
  static const _bodyEdges = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],   // arms
    [5, 11], [6, 12], [11, 12],                  // torso
    [11, 13], [13, 15], [12, 14], [14, 16],      // legs
    [0, 1], [0, 2], [1, 3], [2, 4],             // face
  ];

  // Hand connections (MediaPipe 21-point layout), applied with offset
  static const _handEdges = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // index
    [0, 9], [9, 10], [10, 11], [11, 12],  // middle
    [0, 13], [13, 14], [14, 15], [15, 16],// ring
    [0, 17], [17, 18], [18, 19], [19, 20],// pinky
  ];

  // Minimum confidence to draw a keypoint. SimCC scores are soft-max peaks —
  // valid joints usually score >0.3; noise/occluded joints score near 0.
  static const _minScore = 0.3;

  Offset? _kpOffset(int idx, Size size) {
    if (idx >= keypoints.length) return null;
    final kp = keypoints[idx];
    if (kp[0] == 0.0 && kp[1] == 0.0) return null;
    if (kp[2] < _minScore) return null; // skip low-confidence joints
    // Clamp to [0,1] to guard against occasional out-of-crop coordinates
    final x = kp[0].clamp(0.0, 1.0);
    final y = kp[1].clamp(0.0, 1.0);
    return Offset(x * size.width, y * size.height);
  }

  void _drawEdges(Canvas c, Size s, List<List<int>> edges, int offset, Paint linePaint) {
    for (final e in edges) {
      final a = _kpOffset(e[0] + offset, s);
      final b = _kpOffset(e[1] + offset, s);
      if (a != null && b != null) c.drawLine(a, b, linePaint);
    }
  }

  void _drawDots(Canvas c, Size s, int from, int to, Paint dotPaint, double r) {
    for (int i = from; i < to; i++) {
      final o = _kpOffset(i, s);
      if (o != null) c.drawCircle(o, r, dotPaint);
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (!personDetected || keypoints.isEmpty) return;

    // ── Bounding box from confident keypoints only ────────────────────────── //
    final valid = [
      for (int i = 0; i < keypoints.length; i++)
        if (keypoints[i][2] >= _minScore &&
            (keypoints[i][0] != 0.0 || keypoints[i][1] != 0.0)) keypoints[i]
    ];
    if (valid.isNotEmpty) {
      final xs = valid.map((k) => k[0] * size.width).toList();
      final ys = valid.map((k) => k[1] * size.height).toList();
      final rect = Rect.fromLTRB(
        xs.reduce(min) - 12, ys.reduce(min) - 12,
        xs.reduce(max) + 12, ys.reduce(max) + 12,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        Paint()
          ..color = Colors.greenAccent.withValues(alpha: 0.75)
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke,
      );
    }

    // ── Body skeleton (white lines) ───────────────────────────────────────── //
    final bodyLine = Paint()
      ..color = Colors.white.withValues(alpha: 0.75)
      ..strokeWidth = 1.8
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    _drawEdges(canvas, size, _bodyEdges, 0, bodyLine);

    // Body key joints (green dots)
    _drawDots(canvas, size, 0, 17, Paint()..color = Colors.greenAccent, 4.0);

    // ── Left hand (blue) offset 33 ────────────────────────────────────────── //
    final leftLine = Paint()
      ..color = Colors.lightBlueAccent.withValues(alpha: 0.9)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    _drawEdges(canvas, size, _handEdges, 33, leftLine);
    _drawDots(canvas, size, 33, 54, Paint()..color = Colors.lightBlueAccent, 3.5);

    // ── Right hand (orange) offset 54 ────────────────────────────────────── //
    final rightLine = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.9)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    _drawEdges(canvas, size, _handEdges, 54, rightLine);
    _drawDots(canvas, size, 54, 75, Paint()..color = Colors.orangeAccent, 3.5);
  }

  @override
  bool shouldRepaint(_SkeletonPainter old) =>
      old.keypoints != keypoints || old.personDetected != personDetected;
}

// ── Supporting widgets ────────────────────────────────────────────────────────── //

class _CameraView extends StatelessWidget {
  final web.HTMLVideoElement videoElement;
  const _CameraView({required this.videoElement});

  @override
  Widget build(BuildContext context) {
    return HtmlElementView.fromTagName(
      tagName: 'video',
      onElementCreated: (element) {
        final v = element as web.HTMLVideoElement;
        v.srcObject = videoElement.srcObject;
        v.autoplay = true;
        v.muted = true;
        v.style.width = '100%';
        v.style.height = '100%';
        v.style.objectFit = 'cover';
      },
    );
  }
}

class _SignVideoPlayer extends StatelessWidget {
  final String base64Mp4;
  const _SignVideoPlayer({super.key, required this.base64Mp4});

  @override
  Widget build(BuildContext context) {
    return HtmlElementView.fromTagName(
      tagName: 'video',
      onElementCreated: (element) {
        final v = element as web.HTMLVideoElement;
        v.src = 'data:video/mp4;base64,$base64Mp4';
        v.autoplay = true;
        v.controls = true;
        v.style.width = '100%';
        v.style.height = '100%';
        v.style.objectFit = 'contain';
      },
    );
  }
}

class _PanelHeader extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color, bg;
  const _PanelHeader({required this.icon, required this.label, required this.color, required this.bg});

  @override
  Widget build(BuildContext context) => Container(
    color: bg,
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    child: Row(children: [
      Icon(icon, size: 14, color: color),
      const SizedBox(width: 5),
      Text(label, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
          color: color, letterSpacing: 0.3)),
    ]),
  );
}

class _EmptyState extends StatelessWidget {
  final IconData icon;
  final String label;
  final ColorScheme cs;
  const _EmptyState({required this.icon, required this.label, required this.cs});

  @override
  Widget build(BuildContext context) => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 28, color: cs.outline),
      const SizedBox(height: 8),
      Text(label, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
          textAlign: TextAlign.center),
    ]),
  );
}

class _GlossBubble extends StatelessWidget {
  final _Msg msg;
  final ColorScheme cs;
  const _GlossBubble({required this.msg, required this.cs});

  @override
  Widget build(BuildContext context) {
    final t = '${msg.time.hour.toString().padLeft(2,'0')}:${msg.time.minute.toString().padLeft(2,'0')}';
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        constraints: const BoxConstraints(maxWidth: 340),
        decoration: BoxDecoration(
          color: cs.secondaryContainer,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(4), topRight: Radius.circular(12),
            bottomRight: Radius.circular(12), bottomLeft: Radius.circular(12),
          ),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(msg.text, style: TextStyle(fontFamily: 'monospace', fontSize: 14,
              fontWeight: FontWeight.w600, letterSpacing: 1.2,
              color: cs.onSecondaryContainer)),
          const SizedBox(height: 3),
          Text(t, style: TextStyle(fontSize: 10,
              color: cs.onSecondaryContainer.withValues(alpha: 0.6))),
        ]),
      ),
    );
  }
}

class _RecognitionBubble extends StatelessWidget {
  final _Msg msg;
  final ColorScheme cs;
  final VoidCallback? onPlay;
  const _RecognitionBubble({required this.msg, required this.cs, this.onPlay});

  @override
  Widget build(BuildContext context) {
    final t = '${msg.time.hour.toString().padLeft(2,'0')}:${msg.time.minute.toString().padLeft(2,'0')}';
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.fromLTRB(12, 8, 6, 8),
        constraints: const BoxConstraints(maxWidth: 340),
        decoration: BoxDecoration(
          color: cs.primaryContainer,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(12), topRight: Radius.circular(4),
            bottomRight: Radius.circular(12), bottomLeft: Radius.circular(12),
          ),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(msg.text, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w500,
                color: cs.onPrimaryContainer)),
            const SizedBox(height: 3),
            Text(t, style: TextStyle(fontSize: 10,
                color: cs.onPrimaryContainer.withValues(alpha: 0.6))),
          ])),
          if (onPlay != null)
            IconButton(
              icon: Icon(Icons.volume_up_outlined, size: 18, color: cs.onPrimaryContainer),
              tooltip: 'Озвучить', padding: const EdgeInsets.fromLTRB(4,0,0,0),
              constraints: const BoxConstraints(), onPressed: onPlay,
            ),
        ]),
      ),
    );
  }
}

class _WsDot extends StatelessWidget {
  final _WsStatus status;
  const _WsDot({required this.status});
  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      _WsStatus.connected    => Colors.green,
      _WsStatus.connecting   => Colors.orange,
      _WsStatus.error        => Colors.red,
      _WsStatus.disconnected => Colors.grey,
    };
    return Tooltip(message: 'WS: ${status.name}',
        child: Container(width: 8, height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle)));
  }
}

class _LatencyBadge extends StatelessWidget {
  final int ms;
  const _LatencyBadge({required this.ms});
  @override
  Widget build(BuildContext context) {
    final color = ms <= 500 ? Colors.green : ms <= 2000 ? Colors.orange : Colors.red;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        border: Border.all(color: color.withValues(alpha: 0.4)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text('${ms}ms',
          style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w600)),
    );
  }
}

class _UrlField extends StatefulWidget {
  final String initialUrl;
  final ValueChanged<String> onChanged;
  const _UrlField({required this.initialUrl, required this.onChanged});
  @override
  State<_UrlField> createState() => _UrlFieldState();
}

class _UrlFieldState extends State<_UrlField> {
  late final TextEditingController _ctrl;
  @override void initState() { super.initState(); _ctrl = TextEditingController(text: widget.initialUrl); }
  @override void dispose() { _ctrl.dispose(); super.dispose(); }
  @override
  Widget build(BuildContext context) => TextField(
    controller: _ctrl,
    style: const TextStyle(fontSize: 11),
    decoration: const InputDecoration(
      isDense: true, contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      border: OutlineInputBorder(), hintText: 'ws://host:8000',
      prefixIcon: Icon(Icons.dns_outlined, size: 14),
    ),
    onSubmitted: widget.onChanged,
  );
}
