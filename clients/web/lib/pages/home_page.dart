// ignore_for_file: avoid_web_libraries_in_flutter
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';
import 'package:web/web.dart' as web;
import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';

// ── WebSocket message types ──────────────────────────────────────────────── //

enum _WsStatus { disconnected, connecting, connected, error }

class _GlossItem {
  final String gloss;
  final double prob;
  const _GlossItem(this.gloss, this.prob);
}

// ── HomePage ─────────────────────────────────────────────────────────────── //

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // ── Camera / canvas ───────────────────────────────────────────────────── //
  web.HTMLVideoElement? _video;
  web.HTMLCanvasElement? _canvas;
  web.MediaStream? _mediaStream;
  Timer? _frameTimer;
  bool _cameraActive = false;
  String _cameraError = '';

  // ── WebSocket (RSL → Text) ────────────────────────────────────────────── //
  WebSocketChannel? _rslWs;
  StreamSubscription<dynamic>? _rslSub;
  _WsStatus _rslStatus = _WsStatus.disconnected;
  List<_GlossItem> _glosses = [];
  String _rslResult = '';
  String _rslAudio = ''; // base64 wav

  // ── WebSocket (Text → RSL) ────────────────────────────────────────────── //
  WebSocketChannel? _ttsWs;
  StreamSubscription<dynamic>? _ttsSub;
  _WsStatus _ttsStatus = _WsStatus.disconnected;
  final _textCtrl = TextEditingController();
  String _asrText = '';
  String _glossSequence = '';
  bool _ttsProcessing = false;

  // ── Latency tracking ──────────────────────────────────────────────────── //
  int? _latencyMs;
  DateTime? _lastFrameSent;

  final _sessionId = const Uuid().v4();

  // ── Camera ───────────────────────────────────────────────────────────── //

  Future<void> _startCamera() async {
    try {
      final constraints = web.MediaStreamConstraints(
        video: true.toJS,
        audio: false.toJS,
      );
      final stream =
          await web.window.navigator.mediaDevices.getUserMedia(constraints).toDart;
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
    if (_video == null || _canvas == null) return;
    if (_rslStatus != _WsStatus.connected) return;

    final ctx = _canvas!.getContext('2d') as web.CanvasRenderingContext2D?;
    if (ctx == null) return;

    ctx.drawImage(_video!, 0, 0);
    final dataUrl = _canvas!.toDataURL('image/jpeg', 0.7.toJS);
    final b64 = dataUrl.split(',').last;

    _lastFrameSent = DateTime.now();
    _rslWs!.sink.add(jsonEncode({
      'type': 'video_frame',
      'frame': b64,
      'session_id': _sessionId,
    }));
  }

  // ── RSL → Text WebSocket ──────────────────────────────────────────────── //

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
    } catch (e) {
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

    switch (type) {
      case 'gloss':
        final items = (msg['glosses'] as List<dynamic>? ?? [])
            .map((g) => _GlossItem(
                  g['gloss'] as String,
                  (g['prob'] as num).toDouble(),
                ))
            .toList();
        setState(() {
          _glosses = items;
          if (_lastFrameSent != null) {
            _latencyMs = DateTime.now().difference(_lastFrameSent!).inMilliseconds;
          }
        });
        break;

      case 'result':
        setState(() => _rslResult = msg['text'] as String? ?? '');
        break;

      case 'audio':
        setState(() => _rslAudio = msg['audio'] as String? ?? '');
        _playAudio(_rslAudio);
        break;

      case 'error':
        setState(() => _rslStatus = _WsStatus.error);
        break;
    }
  }

  // ── Text → RSL WebSocket ──────────────────────────────────────────────── //

  Future<void> _translateText() async {
    final text = _textCtrl.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _ttsProcessing = true;
      _asrText = '';
      _glossSequence = '';
      _ttsStatus = _WsStatus.connecting;
    });

    try {
      _ttsWs = WebSocketChannel.connect(Uri.parse(Config.wsTextToRsl));
      await _ttsWs!.ready;
      setState(() => _ttsStatus = _WsStatus.connected);

      _ttsWs!.sink.add(jsonEncode({
        'type': 'text_input',
        'text': text,
        'session_id': _sessionId,
      }));

      await for (final raw in _ttsWs!.stream) {
        final msg = jsonDecode(raw as String) as Map<String, dynamic>;
        final t = msg['type'] as String?;
        if (!mounted) break;
        if (t == 'result') {
          setState(() {
            _glossSequence = msg['text'] as String? ?? '';
            _ttsProcessing = false;
          });
          break;
        } else if (t == 'chunk') {
          setState(() => _asrText = msg['text'] as String? ?? '');
        } else if (t == 'error') {
          setState(() => _ttsProcessing = false);
          break;
        }
      }
      await _ttsWs?.sink.close();
      _ttsWs = null;
      if (mounted) setState(() => _ttsStatus = _WsStatus.disconnected);
    } catch (e) {
      _ttsWs = null;
      if (mounted) {
        setState(() {
          _ttsProcessing = false;
          _ttsStatus = _WsStatus.error;
        });
      }
    }

    // Fallback: use REST endpoint
    if (_glossSequence.isEmpty && mounted) {
      await _translateTextRest(text);
    }
  }

  Future<void> _translateTextRest(String text) async {
    try {
      final uri = Uri.parse('${Config.httpBase}/api/v1/translate');
      final resp = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'mode': 'text_to_rsl',
          'text': text,
          'session_id': _sessionId,
        }),
      );
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      if (mounted) {
        setState(() {
          _glossSequence = data['translation'] as String? ?? '';
          _ttsProcessing = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _ttsProcessing = false);
    }
  }

  // ── Audio playback ────────────────────────────────────────────────────── //

  void _playAudio(String base64Wav) {
    if (base64Wav.isEmpty) return;
    final dataUrl = 'data:audio/wav;base64,$base64Wav';
    final audio = web.HTMLAudioElement()..src = dataUrl;
    audio.play();
  }

  // ── UI ────────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isWide = MediaQuery.of(context).size.width >= 900;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.sign_language, color: cs.primary),
            const SizedBox(width: 8),
            const Text('Glossa'),
          ],
        ),
        actions: [
          _LatencyChip(ms: _latencyMs),
          const SizedBox(width: 8),
          _StatusDot(status: _rslStatus, label: 'RSL→Text'),
          const SizedBox(width: 4),
          _StatusDot(status: _ttsStatus, label: 'Text→RSL'),
          const SizedBox(width: 16),
          // Server URL field
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
      body: isWide ? _wideLayout(cs) : _narrowLayout(cs),
    );
  }

  Widget _wideLayout(ColorScheme cs) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(child: _leftColumn(cs)),
        const VerticalDivider(width: 1),
        Expanded(child: _rightColumn(cs)),
      ],
    );
  }

  Widget _narrowLayout(ColorScheme cs) {
    return SingleChildScrollView(
      child: Column(
        children: [
          _leftColumn(cs),
          const Divider(),
          _rightColumn(cs),
        ],
      ),
    );
  }

  // ── Left column: RSL → Text ─────────────────────────────────────────── //

  Widget _leftColumn(ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Жест → Речь',
              style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),

          // Camera preview area
          AspectRatio(
            aspectRatio: 4 / 3,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: _cameraActive
                  ? _WebCameraPreview(videoElement: _video!)
                  : Container(
                      color: cs.surfaceContainerHighest,
                      child: Center(
                        child: _cameraError.isNotEmpty
                            ? Text(_cameraError,
                                style: TextStyle(color: cs.error))
                            : Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(Icons.videocam_off_outlined,
                                      size: 48, color: cs.outline),
                                  const SizedBox(height: 8),
                                  const Text('Камера не запущена'),
                                ],
                              ),
                      ),
                    ),
            ),
          ),
          const SizedBox(height: 12),

          // Camera toggle button
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

          const SizedBox(height: 16),

          // Gloss results
          if (_glosses.isNotEmpty) ...[
            Text('Топ-3 глоссы',
                style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 8),
            ..._glosses.asMap().entries.map(
                  (e) => _WebGlossRow(item: e.value, rank: e.key + 1, cs: cs),
                ),
            const SizedBox(height: 12),
          ],

          // Translation result
          if (_rslResult.isNotEmpty)
            Card(
              color: cs.primaryContainer,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(_rslResult,
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(color: cs.onPrimaryContainer)),
                    ),
                    IconButton(
                      icon: const Icon(Icons.volume_up_outlined),
                      tooltip: 'Воспроизвести',
                      onPressed: () => _playAudio(_rslAudio),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  // ── Right column: Text → RSL ────────────────────────────────────────── //

  Widget _rightColumn(ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Речь → Жест',
              style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),

          // Text input
          TextField(
            controller: _textCtrl,
            maxLines: 4,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              labelText: 'Введите текст',
              hintText: 'Пример: Я хочу пить воду',
            ),
            onSubmitted: (_) => _translateText(),
          ),
          const SizedBox(height: 12),

          FilledButton.icon(
            onPressed: _ttsProcessing ? null : _translateText,
            icon: const Icon(Icons.translate),
            label: const Text('Перевести в глоссы РЖЯ'),
          ),

          const SizedBox(height: 16),

          // Processing indicator
          if (_ttsProcessing)
            const Center(child: CircularProgressIndicator()),

          // ASR text (for voice input pathway)
          if (_asrText.isNotEmpty) ...[
            const SizedBox(height: 8),
            _InfoCard(
              icon: Icons.hearing,
              label: 'Распознанный текст',
              text: _asrText,
              color: cs.secondaryContainer,
            ),
          ],

          // Gloss sequence output
          if (_glossSequence.isNotEmpty) ...[
            const SizedBox(height: 12),
            _InfoCard(
              icon: Icons.sign_language,
              label: 'Глоссы РЖЯ',
              text: _glossSequence,
              color: cs.primaryContainer,
              monospace: true,
            ),
          ],

          const Spacer(),

          // About note
          Text(
            'Glossa — система перевода русского жестового языка',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
            textAlign: TextAlign.center,
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
    _ttsSub?.cancel();
    _ttsWs?.sink.close();
    _textCtrl.dispose();
    super.dispose();
  }
}

// ── Supporting widgets ────────────────────────────────────────────────────── //

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

class _WebGlossRow extends StatelessWidget {
  final _GlossItem item;
  final int rank;
  final ColorScheme cs;

  const _WebGlossRow({
    required this.item,
    required this.rank,
    required this.cs,
  });

  @override
  Widget build(BuildContext context) {
    final pct = (item.prob * 100).toStringAsFixed(1);
    final isTop = rank == 1;

    return Card(
      elevation: isTop ? 3 : 0,
      color: isTop ? cs.primaryContainer : cs.surfaceContainerLow,
      margin: const EdgeInsets.symmetric(vertical: 3),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: [
            CircleAvatar(
              radius: 12,
              backgroundColor: isTop ? cs.primary : cs.outlineVariant,
              child: Text('$rank',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: isTop ? cs.onPrimary : cs.onSurface,
                  )),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(item.gloss,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
                          )),
                  const SizedBox(height: 3),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: item.prob,
                      minHeight: 6,
                      color: item.prob >= 0.7
                          ? Colors.green
                          : item.prob >= 0.4
                              ? cs.secondary
                              : cs.error,
                      backgroundColor: cs.surfaceContainerHighest,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            Text('$pct%',
                style: Theme.of(context).textTheme.labelMedium?.copyWith(
                      color: cs.onSurfaceVariant,
                    )),
          ],
        ),
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String text;
  final Color color;
  final bool monospace;

  const _InfoCard({
    required this.icon,
    required this.label,
    required this.text,
    required this.color,
    this.monospace = false,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: color,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(icon, size: 16),
              const SizedBox(width: 6),
              Text(label, style: Theme.of(context).textTheme.labelMedium),
            ]),
            const SizedBox(height: 8),
            Text(
              text,
              style: monospace
                  ? Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontFamily: 'monospace', letterSpacing: 1.1)
                  : Theme.of(context).textTheme.bodyLarge,
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusDot extends StatelessWidget {
  final _WsStatus status;
  final String label;
  const _StatusDot({required this.status, required this.label});

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      _WsStatus.connected    => Colors.green,
      _WsStatus.connecting   => Colors.orange,
      _WsStatus.error        => Colors.red,
      _WsStatus.disconnected => Colors.grey,
    };
    return Tooltip(
      message: '$label: ${status.name}',
      child: Container(
        width: 10,
        height: 10,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      ),
    );
  }
}

class _LatencyChip extends StatelessWidget {
  final int? ms;
  const _LatencyChip({this.ms});

  @override
  Widget build(BuildContext context) {
    if (ms == null) return const SizedBox.shrink();
    final color = ms! <= 500
        ? Colors.green
        : ms! <= 2000
            ? Colors.orange
            : Colors.red;
    return Chip(
      label: Text('${ms}ms',
          style: TextStyle(color: color, fontSize: 11)),
      side: BorderSide(color: color.withOpacity(0.5)),
      backgroundColor: color.withOpacity(0.1),
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
