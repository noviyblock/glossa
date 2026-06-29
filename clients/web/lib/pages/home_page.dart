// ignore_for_file: avoid_web_libraries_in_flutter
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';
import 'package:web/web.dart' as web;
import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';

enum _WsStatus { disconnected, connecting, connected, error }

class _GlossItem {
  final String gloss;
  final double prob;
  const _GlossItem(this.gloss, this.prob);
}

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
  List<_GlossItem> _glosses = [];
  String _rslResult = '';
  String _rslAudio = '';
  int? _latencyMs;
  DateTime? _lastFrameSent;

  // ── Text → RSL (REST only) ───────────────────────────────────────────────── //
  final _textCtrl = TextEditingController();
  String _glossSequence = '';
  bool _ttsProcessing = false;

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
      _canvas = web.HTMLCanvasElement()
        ..width = 320
        ..height = 240;
      if (mounted) setState(() => _cameraActive = true);
      await _connectRslWs();
      _frameTimer = Timer.periodic(const Duration(milliseconds: 100), (_) => _sendFrame());
    } catch (e) {
      if (mounted) setState(() => _cameraError = 'Камера: $e');
    }
  }

  void _stopCamera() {
    _frameTimer?.cancel();
    _frameTimer = null;
    _mediaStream?.getTracks().toDart.forEach((t) => t.stop());
    _mediaStream = null;
    _disconnectRslWs();
    if (mounted) {
      setState(() {
        _cameraActive = false;
        _glosses = [];
        _rslResult = '';
      });
    }
  }

  void _sendFrame() {
    if (_video == null || _canvas == null || _rslStatus != _WsStatus.connected) return;
    final ctx = _canvas!.getContext('2d') as web.CanvasRenderingContext2D?;
    if (ctx == null) return;
    ctx.drawImage(_video!, 0, 0);
    final b64 = _canvas!.toDataURL('image/jpeg', 0.7.toJS).split(',').last;
    _lastFrameSent = DateTime.now();
    _rslWs!.sink.add(jsonEncode({'type': 'video_frame', 'frame': b64, 'session_id': _sessionId}));
  }

  // ── RSL → Text WS ────────────────────────────────────────────────────────── //

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
        setState(() {
          _glosses = items;
          if (_lastFrameSent != null) {
            _latencyMs = DateTime.now().difference(_lastFrameSent!).inMilliseconds;
          }
        });
        break;
      case 'result':
        setState(() => _rslResult = payload['text'] as String? ?? '');
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
    setState(() {
      _ttsProcessing = true;
      _glossSequence = '';
    });
    try {
      final resp = await http.post(
        Uri.parse('${Config.httpBase}/api/v1/translate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'mode': 'text_to_rsl', 'text': text, 'session_id': _sessionId}),
      );
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      if (mounted) setState(() => _glossSequence = data['translation'] as String? ?? '');
    } catch (_) {
    } finally {
      if (mounted) setState(() => _ttsProcessing = false);
    }
  }

  void _playAudio(String base64Wav) {
    if (base64Wav.isEmpty) return;
    (web.HTMLAudioElement()..src = 'data:audio/wav;base64,$base64Wav').play();
  }

  // ── Build ─────────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isWide = MediaQuery.of(context).size.width >= 800;

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 12,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.sign_language, color: cs.primary),
            const SizedBox(width: 8),
            const Text('Glossa'),
            const SizedBox(width: 10),
            _StatusDot(status: _rslStatus),
            if (_latencyMs != null) ...[
              const SizedBox(width: 8),
              _LatencyChip(ms: _latencyMs!),
            ],
          ],
        ),
        actions: [
          SizedBox(
            width: 220,
            child: _ServerUrlField(
              initialUrl: Config.serverUrl,
              onChanged: (url) => Config.serverUrl = url,
            ),
          ),
          const SizedBox(width: 12),
        ],
      ),
      body: isWide
          ? Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Expanded(flex: 3, child: _deafPanel(cs)),
              const VerticalDivider(width: 1),
              Expanded(flex: 2, child: _hearingPanel(cs)),
            ])
          : SingleChildScrollView(
              child: Column(children: [
                _deafPanel(cs),
                const Divider(),
                _hearingPanel(cs),
              ]),
            ),
    );
  }

  // ── Левая панель — глухонемой ─────────────────────────────────────────────── //

  Widget _deafPanel(ColorScheme cs) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Title
          Row(children: [
            Icon(Icons.sign_language, size: 18, color: cs.primary),
            const SizedBox(width: 6),
            Text('Показываю жесты',
                style: Theme.of(context).textTheme.titleMedium),
          ]),
          const SizedBox(height: 12),

          // Camera preview
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: AspectRatio(
              aspectRatio: 4 / 3,
              child: _cameraActive
                  ? _WebCameraPreview(videoElement: _video!)
                  : ColoredBox(
                      color: cs.surfaceContainerHighest,
                      child: Center(
                        child: _cameraError.isNotEmpty
                            ? Padding(
                                padding: const EdgeInsets.all(16),
                                child: Text(_cameraError,
                                    style: TextStyle(color: cs.error),
                                    textAlign: TextAlign.center),
                              )
                            : Column(mainAxisSize: MainAxisSize.min, children: [
                                Icon(Icons.videocam_off_outlined,
                                    size: 40, color: cs.outline),
                                const SizedBox(height: 8),
                                Text('Камера не запущена',
                                    style: TextStyle(color: cs.onSurfaceVariant)),
                              ]),
                      ),
                    ),
            ),
          ),
          const SizedBox(height: 10),

          // Camera button
          FilledButton.icon(
            onPressed: _cameraActive ? _stopCamera : _startCamera,
            icon: Icon(_cameraActive ? Icons.stop : Icons.videocam),
            label: Text(_cameraActive ? 'Остановить' : 'Запустить камеру'),
            style: _cameraActive
                ? FilledButton.styleFrom(
                    backgroundColor: cs.error,
                    foregroundColor: cs.onError,
                  )
                : null,
          ),

          // Top-3 glosses as compact chips
          if (_glosses.isNotEmpty) ...[
            const SizedBox(height: 14),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: _glosses.asMap().entries.map((e) {
                final isTop = e.key == 0;
                final pct = (e.value.prob * 100).toStringAsFixed(0);
                return Chip(
                  avatar: CircleAvatar(
                    radius: 10,
                    backgroundColor:
                        isTop ? cs.primary : cs.surfaceContainerHighest,
                    child: Text(
                      '${e.key + 1}',
                      style: TextStyle(
                        fontSize: 10,
                        color: isTop ? cs.onPrimary : cs.onSurfaceVariant,
                      ),
                    ),
                  ),
                  label: Text(
                    '${e.value.gloss}  $pct%',
                    style: TextStyle(
                      fontWeight:
                          isTop ? FontWeight.bold : FontWeight.normal,
                      fontSize: 13,
                    ),
                  ),
                  backgroundColor:
                      isTop ? cs.primaryContainer : cs.surfaceContainerLow,
                  side: BorderSide.none,
                );
              }).toList(),
            ),
          ],

          // Message from hearing person (gloss sequence reply)
          if (_glossSequence.isNotEmpty) ...[
            const SizedBox(height: 20),
            const Divider(),
            const SizedBox(height: 8),
            Row(children: [
              Icon(Icons.chat_bubble_outline, size: 15, color: cs.secondary),
              const SizedBox(width: 6),
              Text('Сообщение от собеседника',
                  style: Theme.of(context)
                      .textTheme
                      .labelMedium
                      ?.copyWith(color: cs.secondary)),
            ]),
            const SizedBox(height: 8),
            Card(
              color: cs.secondaryContainer,
              elevation: 0,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  _glossSequence,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontFamily: 'monospace',
                        letterSpacing: 1.5,
                        color: cs.onSecondaryContainer,
                      ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  // ── Правая панель — слышащий ──────────────────────────────────────────────── //

  Widget _hearingPanel(ColorScheme cs) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Title
          Row(children: [
            Icon(Icons.hearing, size: 18, color: cs.tertiary),
            const SizedBox(width: 6),
            Text('Читаю жесты',
                style: Theme.of(context).textTheme.titleMedium),
          ]),
          const SizedBox(height: 12),

          // Recognized gesture text — main output for hearing person
          _rslResult.isNotEmpty
              ? Card(
                  color: cs.primaryContainer,
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 14, 8, 14),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: Text(
                            _rslResult,
                            style: Theme.of(context)
                                .textTheme
                                .headlineSmall
                                ?.copyWith(color: cs.onPrimaryContainer),
                          ),
                        ),
                        if (_rslAudio.isNotEmpty)
                          IconButton(
                            icon: const Icon(Icons.volume_up_outlined),
                            tooltip: 'Озвучить',
                            onPressed: () => _playAudio(_rslAudio),
                          ),
                      ],
                    ),
                  ),
                )
              : Card(
                  color: cs.surfaceContainerLow,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Row(children: [
                      Icon(Icons.sign_language,
                          color: cs.outline, size: 28),
                      const SizedBox(width: 12),
                      Text('Жесты появятся здесь',
                          style:
                              TextStyle(color: cs.onSurfaceVariant)),
                    ]),
                  ),
                ),

          const SizedBox(height: 24),
          const Divider(),
          const SizedBox(height: 12),

          // Text reply input
          Row(children: [
            Icon(Icons.edit_outlined, size: 15, color: cs.tertiary),
            const SizedBox(width: 6),
            Text('Написать ответ',
                style: Theme.of(context)
                    .textTheme
                    .labelMedium
                    ?.copyWith(color: cs.tertiary)),
          ]),
          const SizedBox(height: 8),

          TextField(
            controller: _textCtrl,
            maxLines: 3,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: 'Введите текст на русском…',
            ),
            onSubmitted: (_) => _sendText(),
          ),
          const SizedBox(height: 8),

          FilledButton.icon(
            onPressed: _ttsProcessing ? null : _sendText,
            icon: _ttsProcessing
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.send),
            label: const Text('Отправить'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _stopCamera();
    _rslSub?.cancel();
    _rslWs?.sink.close();
    _textCtrl.dispose();
    super.dispose();
  }
}

// ── Supporting widgets ────────────────────────────────────────────────────────── //

class _WebCameraPreview extends StatelessWidget {
  final web.HTMLVideoElement videoElement;
  const _WebCameraPreview({required this.videoElement});

  @override
  Widget build(BuildContext context) {
    return HtmlElementView.fromTagName(
      tagName: 'video',
      onElementCreated: (element) {
        final video = element as web.HTMLVideoElement;
        video.srcObject = videoElement.srcObject;
        video.autoplay = true;
        video.muted = true;
        video.style.width = '100%';
        video.style.height = '100%';
        video.style.objectFit = 'cover';
      },
    );
  }
}

class _StatusDot extends StatelessWidget {
  final _WsStatus status;
  const _StatusDot({required this.status});

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      _WsStatus.connected    => Colors.green,
      _WsStatus.connecting   => Colors.orange,
      _WsStatus.error        => Colors.red,
      _WsStatus.disconnected => Colors.grey,
    };
    return Tooltip(
      message: 'WS: ${status.name}',
      child: Container(
        width: 10,
        height: 10,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      ),
    );
  }
}

class _LatencyChip extends StatelessWidget {
  final int ms;
  const _LatencyChip({required this.ms});

  @override
  Widget build(BuildContext context) {
    final color = ms <= 500
        ? Colors.green
        : ms <= 2000
            ? Colors.orange
            : Colors.red;
    return Chip(
      label: Text('${ms}ms',
          style: TextStyle(color: color, fontSize: 11)),
      side: BorderSide(color: color.withValues(alpha: 0.4)),
      backgroundColor: color.withValues(alpha: 0.08),
      padding: EdgeInsets.zero,
      visualDensity: VisualDensity.compact,
    );
  }
}

class _ServerUrlField extends StatefulWidget {
  final String initialUrl;
  final ValueChanged<String> onChanged;
  const _ServerUrlField({required this.initialUrl, required this.onChanged});

  @override
  State<_ServerUrlField> createState() => _ServerUrlFieldState();
}

class _ServerUrlFieldState extends State<_ServerUrlField> {
  late final TextEditingController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(text: widget.initialUrl);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: _ctrl,
      style: const TextStyle(fontSize: 12),
      decoration: const InputDecoration(
        isDense: true,
        contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        border: OutlineInputBorder(),
        hintText: 'ws://localhost:8000',
        prefixIcon: Icon(Icons.dns_outlined, size: 16),
      ),
      onSubmitted: widget.onChanged,
    );
  }
}
