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
  List<_GlossItem> _liveGlosses = []; // top-3 overlay on camera
  String _rslAudio = '';
  int? _latencyMs;
  DateTime? _lastFrameSent;

  // RSL recognition chat (panel 3)
  final List<_Msg> _rslMsgs = [];
  final _rslScrollCtrl = ScrollController();
  String? _lastRslText;
  DateTime? _lastRslAt;

  // ── Text → RSL REST ───────────────────────────────────────────────────────── //
  final _textCtrl = TextEditingController();
  bool _ttsProcessing = false;

  // Gloss chat (panel 2)
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
      _canvas = web.HTMLCanvasElement()..width = 320..height = 240;
      if (mounted) setState(() => _cameraActive = true);
      await _connectRslWs();
      _frameTimer = Timer.periodic(const Duration(milliseconds: 100), (_) => _sendFrame());
    } catch (e) {
      if (mounted) setState(() => _cameraError = '$e');
    }
  }

  void _stopCamera() {
    _frameTimer?.cancel();
    _frameTimer = null;
    _mediaStream?.getTracks().toDart.forEach((t) => t.stop());
    _mediaStream = null;
    _disconnectRslWs();
    if (mounted) setState(() { _cameraActive = false; _liveGlosses = []; });
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
        setState(() {
          _liveGlosses = items;
          if (_lastFrameSent != null) {
            _latencyMs = DateTime.now().difference(_lastFrameSent!).inMilliseconds;
          }
        });
        break;

      case 'result':
        final text = payload['text'] as String? ?? '';
        if (text.isEmpty) break;
        // Дедупликация: не добавлять если то же слово меньше чем 3с назад
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
        ctrl.animateTo(
          ctrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final w = MediaQuery.of(context).size.width;
    final isWide = w >= 800;

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
      body: isWide ? _wideLayout(cs) : _narrowLayout(cs),
    );
  }

  // ── Wide layout: 4 panels in a 2×2 grid ──────────────────────────────────── //

  Widget _wideLayout(ColorScheme cs) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Left half
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Panel 1: Camera
              Flexible(flex: 5, child: _cameraPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              // Panel 2: Gloss chat
              Expanded(flex: 5, child: _glossChatPanel(cs)),
            ],
          ),
        ),
        VerticalDivider(width: 1, color: cs.outlineVariant),
        // Right half
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Panel 3: Recognition chat
              Expanded(child: _recognitionChatPanel(cs)),
              Divider(height: 1, color: cs.outlineVariant),
              // Panel 4: Text input
              _textInputPanel(cs),
            ],
          ),
        ),
      ],
    );
  }

  // ── Narrow layout: 4 panels stacked ──────────────────────────────────────── //

  Widget _narrowLayout(ColorScheme cs) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(height: 220, child: _cameraPanel(cs)),
        Divider(height: 1, color: cs.outlineVariant),
        SizedBox(height: 260, child: _glossChatPanel(cs)),
        Divider(height: 1, color: cs.outlineVariant),
        SizedBox(height: 260, child: _recognitionChatPanel(cs)),
        Divider(height: 1, color: cs.outlineVariant),
        _textInputPanel(cs),
      ],
    );
  }

  // ── Panel 1: Camera ───────────────────────────────────────────────────────── //

  Widget _cameraPanel(ColorScheme cs) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Video or placeholder
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
                            Text('Камера',
                                style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
                          ]),
                  ),
                ),
        ),

        // Live gloss chips overlay
        if (_liveGlosses.isNotEmpty)
          Positioned(
            top: 6, left: 6, right: 6,
            child: Wrap(
              spacing: 4,
              runSpacing: 4,
              children: _liveGlosses.asMap().entries.map((e) {
                final isTop = e.key == 0;
                final pct = (e.value.prob * 100).toStringAsFixed(0);
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: isTop
                        ? cs.primary.withValues(alpha: 0.88)
                        : Colors.black54,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${e.value.gloss} $pct%',
                    style: TextStyle(
                      color: isTop ? cs.onPrimary : Colors.white,
                      fontSize: 12,
                      fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                );
              }).toList(),
            ),
          ),

        // Camera start/stop button at bottom
        Positioned(
          bottom: 8, left: 0, right: 0,
          child: Center(
            child: FilledButton.icon(
              onPressed: _cameraActive ? _stopCamera : _startCamera,
              icon: Icon(_cameraActive ? Icons.stop : Icons.videocam, size: 16),
              label: Text(
                _cameraActive ? 'Стоп' : 'Камера',
                style: const TextStyle(fontSize: 13),
              ),
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
          ),
        ),
      ],
    );
  }

  // ── Panel 2: Gloss chat (text → RSL) ─────────────────────────────────────── //

  Widget _glossChatPanel(ColorScheme cs) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _PanelHeader(
          icon: Icons.sign_language,
          label: 'Глоссы от собеседника',
          color: cs.secondary,
          bg: cs.secondaryContainer.withValues(alpha: 0.5),
        ),
        Expanded(
          child: _glossMsgs.isEmpty
              ? _EmptyState(
                  icon: Icons.chat_bubble_outline,
                  label: 'Глоссы появятся когда\nсобеседник напишет сообщение',
                  cs: cs,
                )
              : ListView.builder(
                  controller: _glossScrollCtrl,
                  padding: const EdgeInsets.fromLTRB(10, 6, 10, 10),
                  itemCount: _glossMsgs.length,
                  itemBuilder: (_, i) => _GlossBubble(msg: _glossMsgs[i], cs: cs),
                ),
        ),
      ],
    );
  }

  // ── Panel 3: Recognition chat (RSL → text) ────────────────────────────────── //

  Widget _recognitionChatPanel(ColorScheme cs) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _PanelHeader(
          icon: Icons.hearing,
          label: 'Распознанные жесты',
          color: cs.tertiary,
          bg: cs.tertiaryContainer.withValues(alpha: 0.5),
        ),
        Expanded(
          child: _rslMsgs.isEmpty
              ? _EmptyState(
                  icon: Icons.sign_language,
                  label: 'Здесь появится перевод жестов\nпосле запуска камеры',
                  cs: cs,
                )
              : ListView.builder(
                  controller: _rslScrollCtrl,
                  padding: const EdgeInsets.fromLTRB(10, 6, 10, 10),
                  itemCount: _rslMsgs.length,
                  itemBuilder: (_, i) => _RecognitionBubble(
                    msg: _rslMsgs[i],
                    cs: cs,
                    onPlay: _rslAudio.isNotEmpty && i == _rslMsgs.length - 1
                        ? () => _playAudio(_rslAudio)
                        : null,
                  ),
                ),
        ),
      ],
    );
  }

  // ── Panel 4: Text input ───────────────────────────────────────────────────── //

  Widget _textInputPanel(ColorScheme cs) {
    return Container(
      constraints: const BoxConstraints(minHeight: 130, maxHeight: 180),
      color: cs.surface,
      padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: TextField(
              controller: _textCtrl,
              maxLines: null,
              expands: true,
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
                  ? const SizedBox(
                      width: 14, height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.send, size: 16),
              label: const Text('Отправить', style: TextStyle(fontSize: 13)),
            ),
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
    _rslScrollCtrl.dispose();
    _glossScrollCtrl.dispose();
    super.dispose();
  }
}

// ── Reusable widgets ──────────────────────────────────────────────────────────── //

class _PanelHeader extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final Color bg;
  const _PanelHeader({required this.icon, required this.label, required this.color, required this.bg});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: bg,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      child: Row(children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 5),
        Text(label,
            style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: color,
                letterSpacing: 0.3)),
      ]),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final IconData icon;
  final String label;
  final ColorScheme cs;
  const _EmptyState({required this.icon, required this.label, required this.cs});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 28, color: cs.outline),
        const SizedBox(height: 8),
        Text(label,
            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
            textAlign: TextAlign.center),
      ]),
    );
  }
}

class _GlossBubble extends StatelessWidget {
  final _Msg msg;
  final ColorScheme cs;
  const _GlossBubble({required this.msg, required this.cs});

  @override
  Widget build(BuildContext context) {
    final h = msg.time.hour.toString().padLeft(2, '0');
    final m = msg.time.minute.toString().padLeft(2, '0');
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        constraints: const BoxConstraints(maxWidth: 340),
        decoration: BoxDecoration(
          color: cs.secondaryContainer,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(4),
            topRight: Radius.circular(12),
            bottomRight: Radius.circular(12),
            bottomLeft: Radius.circular(12),
          ),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(msg.text,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 14,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.2,
                color: cs.onSecondaryContainer,
              )),
          const SizedBox(height: 3),
          Text('$h:$m',
              style: TextStyle(fontSize: 10, color: cs.onSecondaryContainer.withValues(alpha: 0.6))),
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
    final h = msg.time.hour.toString().padLeft(2, '0');
    final m = msg.time.minute.toString().padLeft(2, '0');
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.fromLTRB(12, 8, 6, 8),
        constraints: const BoxConstraints(maxWidth: 340),
        decoration: BoxDecoration(
          color: cs.primaryContainer,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(12),
            topRight: Radius.circular(4),
            bottomRight: Radius.circular(12),
            bottomLeft: Radius.circular(12),
          ),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(msg.text,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                    color: cs.onPrimaryContainer,
                  )),
              const SizedBox(height: 3),
              Text('$h:$m',
                  style: TextStyle(
                      fontSize: 10,
                      color: cs.onPrimaryContainer.withValues(alpha: 0.6))),
            ]),
          ),
          if (onPlay != null)
            IconButton(
              icon: Icon(Icons.volume_up_outlined, size: 18, color: cs.onPrimaryContainer),
              tooltip: 'Озвучить',
              padding: const EdgeInsets.fromLTRB(4, 0, 0, 0),
              constraints: const BoxConstraints(),
              onPressed: onPlay,
            ),
        ]),
      ),
    );
  }
}

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
    return Tooltip(
      message: 'WS: ${status.name}',
      child: Container(
        width: 8, height: 8,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      ),
    );
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
      style: const TextStyle(fontSize: 11),
      decoration: const InputDecoration(
        isDense: true,
        contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        border: OutlineInputBorder(),
        hintText: 'ws://host:8000',
        prefixIcon: Icon(Icons.dns_outlined, size: 14),
      ),
      onSubmitted: widget.onChanged,
    );
  }
}
