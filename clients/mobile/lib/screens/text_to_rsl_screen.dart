import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:record/record.dart';
import 'package:uuid/uuid.dart';

import '../config.dart';
import '../services/websocket_service.dart';

class TextToRslScreen extends StatefulWidget {
  const TextToRslScreen({super.key});

  @override
  State<TextToRslScreen> createState() => _TextToRslScreenState();
}

class _TextToRslScreenState extends State<TextToRslScreen> {
  final _textController = TextEditingController();
  final _recorder       = AudioRecorder();
  final _player         = AudioPlayer();
  final _sessionId      = const Uuid().v4();

  bool _isRecording     = false;
  bool _isProcessing    = false;
  String _asrText       = '';
  String _glossSequence = '';
  String _statusText    = '';

  final List<StreamSubscription> _subs = [];
  StreamSubscription<Uint8List>? _audioStreamSub;

  @override
  void initState() {
    super.initState();
    _subscribeWs();
  }

  void _subscribeWs() {
    final ws = context.read<WebSocketService>();
    _subs.addAll([
      ws.onChunk.listen((t) {
        if (mounted) setState(() => _asrText = t);
      }),
      ws.onResult.listen((payload) {
        if (mounted) {
          setState(() {
            _glossSequence = payload['text'] as String? ?? '';
            _isProcessing  = false;
            _statusText    = '';
          });
        }
      }),
      ws.onAudio.listen(_playAudio),
      ws.onError.listen((msg) {
        if (mounted) {
          setState(() {
            _statusText   = '⚠ $msg';
            _isProcessing = false;
          });
        }
      }),
    ]);
  }

  // ── Text translation ───────────────────────────────────────────────────── //

  Future<void> _translateText() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    final ws = context.read<WebSocketService>();
    setState(() {
      _isProcessing  = true;
      _statusText    = 'Перевод…';
      _glossSequence = '';
    });

    try {
      if (ws.status != WsStatus.connected) {
        await ws.connect(Config.wsTextToRsl);
      }
      // Encode text as fake "audio" — server handles REST fallback
      // For text input, we call the REST endpoint instead via HTTP
      await _translateTextViaRest(text);
    } catch (e) {
      if (mounted) setState(() => _statusText = 'Ошибка: $e');
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _translateTextViaRest(String text) async {
    // Use HTTP REST for synchronous text→RSL translation
    try {
      final uri = Uri.parse('${Config.httpBase}/api/v1/translate');

      // Simple HTTP POST using Dart's HttpClient
      final client  = HttpClient();
      final request = await client.postUrl(uri);
      request.headers.set('Content-Type', 'application/json');
      request.write(jsonEncode({
        'mode': 'text_to_rsl',
        'text': text,
        'session_id': _sessionId,
      }));
      final resp = await request.close();
      final body = await resp.transform(utf8.decoder).join();
      client.close();

      if (resp.statusCode == 200) {
        final data = jsonDecode(body) as Map<String, dynamic>;
        if (mounted) {
          setState(() {
            _glossSequence = data['translation'] as String? ?? '';
            final wavB64   = data['audio_wav'] as String?;
            if (wavB64 != null && wavB64.isNotEmpty) _playAudio(wavB64);
          });
        }
      }
    } catch (e) {
      if (mounted) setState(() => _statusText = 'REST ошибка: $e');
    }
  }

  // ── Microphone recording ───────────────────────────────────────────────── //

  Future<void> _toggleMicrophone() async {
    if (_isRecording) {
      await _stopRecording();
      return;
    }
    await _startRecording();
  }

  Future<void> _startRecording() async {
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      if (mounted) setState(() => _statusText = 'Нет разрешения на микрофон');
      return;
    }

    final ws = context.read<WebSocketService>();
    try {
      if (ws.status != WsStatus.connected) {
        await ws.connect(Config.wsTextToRsl);
      }
    } catch (e) {
      if (mounted) setState(() => _statusText = 'Ошибка: $e');
      return;
    }

    setState(() {
      _isRecording   = true;
      _asrText       = '';
      _glossSequence = '';
      _statusText    = 'Говорите…';
    });

    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      ),
    );

    _audioStreamSub = stream.listen((chunk) {
      final b64 = base64Encode(chunk);
      ws.sendAudioChunk(b64, _sessionId);
    });
  }

  Future<void> _stopRecording() async {
    _audioStreamSub?.cancel();
    _audioStreamSub = null;
    await _recorder.stop();

    final ws = context.read<WebSocketService>();
    ws.endSession(_sessionId);
    ws.disconnect();

    setState(() {
      _isRecording  = false;
      _isProcessing = true;
      _statusText   = 'Распознавание…';
    });
  }

  // ── TTS playback ───────────────────────────────────────────────────────── //

  Future<void> _playAudio(String base64Wav) async {
    try {
      final bytes = base64Decode(base64Wav);
      final dir   = await getTemporaryDirectory();
      final file  = File('${dir.path}/tts_play.wav');
      await file.writeAsBytes(bytes);
      await _player.setFilePath(file.path);
      await _player.play();
    } catch (e) {
      debugPrint('Playback error: $e');
    }
  }

  // ── UI ─────────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Речь → Жест')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Text input card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    TextField(
                      controller: _textController,
                      maxLines: 3,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        hintText: 'Введите текст для перевода в РЖЯ…',
                        labelText: 'Текст',
                      ),
                    ),
                    const SizedBox(height: 12),
                    FilledButton.icon(
                      onPressed: _isProcessing ? null : _translateText,
                      icon: const Icon(Icons.translate),
                      label: const Text('Перевести в глоссы'),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // OR divider
            Row(
              children: [
                const Expanded(child: Divider()),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: Text('или',
                      style: Theme.of(context).textTheme.labelMedium),
                ),
                const Expanded(child: Divider()),
              ],
            ),

            const SizedBox(height: 8),

            // Microphone button
            Center(
              child: GestureDetector(
                onTap: _toggleMicrophone,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _isRecording ? cs.error : cs.secondaryContainer,
                    boxShadow: _isRecording
                        ? [
                            BoxShadow(
                                color: cs.error.withOpacity(0.4),
                                blurRadius: 20,
                                spreadRadius: 4)
                          ]
                        : [],
                  ),
                  child: Icon(
                    _isRecording ? Icons.stop : Icons.mic,
                    size: 36,
                    color: _isRecording ? cs.onError : cs.onSecondaryContainer,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              _isRecording ? 'Нажмите, чтобы остановить' : 'Нажмите, чтобы говорить',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall,
            ),

            const SizedBox(height: 24),

            // Status / progress
            if (_isProcessing)
              const Center(child: CircularProgressIndicator())
            else if (_statusText.isNotEmpty)
              Text(_statusText,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodySmall),

            // ASR result
            if (_asrText.isNotEmpty) ...[
              const SizedBox(height: 16),
              _ResultCard(
                icon: Icons.hearing,
                label: 'Распознанный текст (ASR)',
                text: _asrText,
                color: cs.secondaryContainer,
              ),
            ],

            // Gloss sequence
            if (_glossSequence.isNotEmpty) ...[
              const SizedBox(height: 12),
              _ResultCard(
                icon: Icons.sign_language,
                label: 'Глоссы РЖЯ',
                text: _glossSequence,
                color: cs.primaryContainer,
                monospace: true,
              ),
            ],

            const SizedBox(height: 80),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    for (final s in _subs) {
      s.cancel();
    }
    _audioStreamSub?.cancel();
    _recorder.dispose();
    _player.dispose();
    _textController.dispose();
    super.dispose();
  }
}

class _ResultCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String text;
  final Color color;
  final bool monospace;

  const _ResultCard({
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
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 18),
                const SizedBox(width: 8),
                Text(label,
                    style: Theme.of(context).textTheme.labelMedium),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              text,
              style: monospace
                  ? Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontFamily: 'monospace', letterSpacing: 1.2)
                  : Theme.of(context).textTheme.bodyLarge,
            ),
          ],
        ),
      ),
    );
  }
}
