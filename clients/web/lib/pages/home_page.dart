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

  // ── RSL → Text WS ────────────────────────────────────────────────────────── //
  WebSocketChannel? _rslWs;
  StreamSubscription<dynamic>? _rslSub;
  _WsStatus _rslStatus = _WsStatus.disconnected;
  List<_GlossItem> _liveGlosses = [];
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

  // RSL recognition chat (panel 3)
  final List<_Msg> _rslMsgs = [];
  final _rslScrollCtrl = ScrollController();
  String? _lastRslText;
  DateTime? _lastRslAt;

  // ── Text → RSL REST ───────────────────────────────────────────────────────── //
  final _textCtrl = TextEditingController();
  bool _ttsProcessing = false;
  final List<_Msg> _glossMsgs = [];
  final _glossScrollCtrl = ScrollController();

  final _sessionId = const Uuid().v4();

  // ── Camera ───────────────────────────────────────────────────────────────── //

  Future<void> _startCamera() async {
    try {
      final stream = await web.window.navigator.mediaDevices
          .getUserMedia(web.MediaStreamConstraints(video: true.toJS, audio: false.toJS))
          .toDart;
      _mediaStream = stream;
      _video = web.HTMLVideoElement()
        ..srcObject = stream
        ..autoplay = true
        ..muted = true;
      // Off-screen video must be explicitly played in some browsers
      try { await _video!.play().toDart; } catch (_) {}
      _canvas = web.HTMLCanvasElement()..width = 640..height = 480;
      if (mounted) setState(() => _cameraActive = true);
      await _connectRslWs();
      _frameTimer = Timer.periodic(const Duration(milliseconds: 100), (_) => _sendFrame());
      // 30fps animation tick: smoothly interpolate skeleton toward latest keypoints
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
    _disconnectRslWs();
    if (mounted) setState(() {
      _cameraActive = false;
      _liveGlosses = [];
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
    // Draw scaled to canvas size so the full frame is captured (not just top-left crop)
    ctx.drawImage(_video!, 0, 0, _canvas!.width, _canvas!.height);
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
        // Skeleton holdout: keep showing 500ms after person disappears (avoids flicker)
        if (personDetected) {
          _hideSkeletonTimer?.cancel();
          _hideSkeletonTimer = null;
          _showSkeleton = true;
          if (kp != null) _targetKp = kp;
        } else if (_showSkeleton && _hideSkeletonTimer == null) {
          _hideSkeletonTimer = Timer(const Duration(milliseconds: 500), () {
            if (mounted) setState(() { _showSkeleton = false; _targetKp = null; _displayKp = null; });
            _hideSkeletonTimer = null;
          });
        }

        setState(() {
          _waitingForResponse = false; // Unblock next frame
          _gestureActive = gestureActive;
          if (items.isNotEmpty) _liveGlosses = items;
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

      case 'audio':
        setState(() => _rslAudio = payload['wav'] as String? ?? '');
        _playAudio(_rslAudio);
        break;

      case 'error':
        setState(() => _rslStatus = _WsStatus.error);
        break;
    }
  }

  // ── Text → RSL REST ──────────────────────────────────────────────────────── //

  Future<void> _sendText() async {
    final text = _textCtrl.text.trim();
    if (text.isEmpty) return;
    _textCtrl.clear();
    setState(() => _ttsProcessing = true);
    try {
      final resp = await http.post(
        Uri.parse('${Config.httpBase}/api/v1/translate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'mode': 'text_to_rsl', 'text': text, 'session_id': _sessionId}),
      );
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      final glosses = data['translation'] as String? ?? '';
      if (glosses.isNotEmpty && mounted) {
        setState(() => _glossMsgs.add(_Msg(glosses)));
        _scrollToBottom(_glossScrollCtrl);
      }
    } catch (_) {
    } finally {
      if (mounted) setState(() => _ttsProcessing = false);
    }
  }

  void _playAudio(String base64Wav) {
    if (base64Wav.isEmpty) return;
    (web.HTMLAudioElement()..src = 'data:audio/wav;base64,$base64Wav').play();
  }

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
      body: isWide
          ? Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Expanded(flex: 3, child: _leftColumn(cs)),
              VerticalDivider(width: 1, color: cs.outlineVariant),
              Expanded(flex: 2, child: _rightColumn(cs)),
            ])
          : Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              SizedBox(height: 220, child: _cameraPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              SizedBox(height: 240, child: _glossChatPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              SizedBox(height: 240, child: _recognitionChatPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              _textInputPanel(cs),
            ]),
    );
  }

  Widget _leftColumn(ColorScheme cs) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Flexible(flex: 5, child: _cameraPanel(cs)),
      Divider(height: 1, color: cs.outlineVariant),
      Expanded(flex: 5, child: _glossChatPanel(cs)),
    ],
  );

  Widget _rightColumn(ColorScheme cs) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Expanded(child: _recognitionChatPanel(cs)),
      Divider(height: 1, color: cs.outlineVariant),
      _textInputPanel(cs),
    ],
  );

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

        // ── Skeleton overlay (animated at 30fps via _animTimer) ──────────── //
        if (_cameraActive && _showSkeleton && _displayKp != null)
          CustomPaint(
            painter: _SkeletonPainter(
              keypoints: _displayKp!,
              personDetected: _showSkeleton,
            ),
          ),

        // ── Live gloss chips (top of frame) ──────────────────────────────── //
        if (_liveGlosses.isNotEmpty)
          Positioned(
            top: 6, left: 6, right: 6,
            child: Wrap(
              spacing: 4, runSpacing: 4,
              children: _liveGlosses.asMap().entries.map((e) {
                final isTop = e.key == 0;
                final pct = (e.value.prob * 100).toStringAsFixed(0);
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: isTop ? cs.primary.withValues(alpha: 0.88) : Colors.black54,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text('${e.value.gloss} $pct%',
                      style: TextStyle(
                        color: isTop ? cs.onPrimary : Colors.white,
                        fontSize: 12,
                        fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
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

  // ── Panel 2: Gloss chat ───────────────────────────────────────────────────── //

  Widget _glossChatPanel(ColorScheme cs) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      _PanelHeader(icon: Icons.sign_language, label: 'Глоссы от собеседника',
          color: cs.secondary, bg: cs.secondaryContainer.withValues(alpha: 0.5)),
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
      SizedBox(
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
    ]),
  );

  @override
  void dispose() {
    _stopCamera();
    _animTimer?.cancel();
    _hideSkeletonTimer?.cancel();
    _rslSub?.cancel();
    _rslWs?.sink.close();
    _textCtrl.dispose();
    _rslScrollCtrl.dispose();
    _glossScrollCtrl.dispose();
    super.dispose();
  }
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
