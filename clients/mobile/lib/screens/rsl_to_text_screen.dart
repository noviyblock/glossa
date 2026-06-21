import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:uuid/uuid.dart';

import '../config.dart';
import '../services/camera_service.dart';
import '../services/websocket_service.dart';
import '../widgets/gloss_card.dart';

class RslToTextScreen extends StatefulWidget {
  const RslToTextScreen({super.key});

  @override
  State<RslToTextScreen> createState() => _RslToTextScreenState();
}

class _RslToTextScreenState extends State<RslToTextScreen> {
  final _camera     = CameraService();
  final _player     = AudioPlayer();
  final _sessionId  = const Uuid().v4();

  bool _recording    = false;
  bool _loading      = true;
  String _statusText = '';
  String _resultText = '';
  List<GlossItem> _glosses = [];

  final List<StreamSubscription> _subs = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
    _subscribeWs();
  }

  Future<void> _initCamera() async {
    try {
      await _camera.initialize();
    } catch (e) {
      if (mounted) {
        setState(() => _statusText = 'Ошибка камеры: $e');
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _subscribeWs() {
    final ws = context.read<WebSocketService>();
    _subs.addAll([
      ws.onGloss.listen((g) {
        if (mounted) setState(() => _glosses = g);
      }),
      ws.onChunk.listen((t) {
        if (mounted) setState(() => _statusText = t);
      }),
      ws.onResult.listen((payload) {
        if (mounted) {
          setState(() {
            _resultText = payload['text'] as String? ?? '';
            _statusText = '';
          });
        }
      }),
      ws.onAudio.listen(_playAudio),
      ws.onError.listen((msg) {
        if (mounted) setState(() => _statusText = '⚠ $msg');
      }),
    ]);
  }

  // ── Recording ──────────────────────────────────────────────────────────── //

  Future<void> _toggleRecording() async {
    final ws = context.read<WebSocketService>();

    if (_recording) {
      _camera.stopStreaming();
      ws.endSession(_sessionId);
      ws.disconnect();
      setState(() => _recording = false);
      return;
    }

    // Connect
    try {
      setState(() => _statusText = 'Подключение…');
      await ws.connect(Config.wsRslToText);
    } catch (e) {
      setState(() => _statusText = 'Ошибка: $e');
      return;
    }

    // Start streaming frames
    _camera.startStreaming((base64Frame) {
      ws.sendVideoFrame(base64Frame, _sessionId);
    });

    setState(() {
      _recording  = true;
      _glosses    = [];
      _resultText = '';
      _statusText = 'Запись…';
    });
  }

  // ── TTS playback ───────────────────────────────────────────────────────── //

  Future<void> _playAudio(String base64Wav) async {
    try {
      final bytes   = base64Decode(base64Wav);
      final dir     = await getTemporaryDirectory();
      final file    = File('${dir.path}/tts_response.wav');
      await file.writeAsBytes(bytes);
      await _player.setFilePath(file.path);
      await _player.play();
    } catch (e) {
      debugPrint('Audio playback error: $e');
    }
  }

  Future<void> _replayTts() async {
    try {
      await _player.seek(Duration.zero);
      await _player.play();
    } catch (_) {}
  }

  // ── UI ─────────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Жест → Речь'),
        actions: [
          IconButton(
            icon: const Icon(Icons.flip_camera_ios_outlined),
            onPressed: () async {
              await _camera.switchCamera();
              if (mounted) setState(() {});
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Camera preview
          Expanded(
            flex: 3,
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _camera.isInitialized
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(_camera.controller!),
                          if (_recording)
                            Positioned(
                              top: 12,
                              right: 12,
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 4),
                                decoration: BoxDecoration(
                                  color: Colors.red,
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: const Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(Icons.circle,
                                        color: Colors.white, size: 10),
                                    SizedBox(width: 4),
                                    Text('REC',
                                        style: TextStyle(
                                            color: Colors.white,
                                            fontSize: 12,
                                            fontWeight: FontWeight.bold)),
                                  ],
                                ),
                              ),
                            ),
                        ],
                      )
                    : Center(
                        child: Text(_statusText.isNotEmpty
                            ? _statusText
                            : 'Камера недоступна'),
                      ),
          ),

          // Results panel
          Expanded(
            flex: 2,
            child: Container(
              color: cs.surface,
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Glosses
                    if (_glosses.isNotEmpty) ...[
                      Text('Топ-3 глоссы',
                          style: Theme.of(context).textTheme.labelLarge),
                      const SizedBox(height: 8),
                      ..._glosses.asMap().entries.map(
                            (e) => GlossCard(item: e.value, rank: e.key + 1),
                          ),
                      const SizedBox(height: 12),
                    ],

                    // Translation result
                    if (_resultText.isNotEmpty)
                      Card(
                        color: cs.primaryContainer,
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Row(
                            children: [
                              Expanded(
                                child: Text(
                                  _resultText,
                                  style: Theme.of(context)
                                      .textTheme
                                      .titleMedium
                                      ?.copyWith(
                                          color: cs.onPrimaryContainer),
                                ),
                              ),
                              IconButton(
                                icon: const Icon(Icons.volume_up_outlined),
                                tooltip: 'Воспроизвести',
                                onPressed: _replayTts,
                              ),
                            ],
                          ),
                        ),
                      ),

                    // Status text
                    if (_statusText.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(
                          _statusText,
                          style: Theme.of(context).textTheme.bodySmall,
                          textAlign: TextAlign.center,
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),

      // Start/Stop FAB
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _loading ? null : _toggleRecording,
        backgroundColor:
            _recording ? cs.errorContainer : cs.primaryContainer,
        icon: Icon(_recording ? Icons.stop : Icons.videocam),
        label: Text(_recording ? 'Стоп' : 'Начать'),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  @override
  void dispose() {
    for (final s in _subs) {
      s.cancel();
    }
    _camera.dispose();
    _player.dispose();
    super.dispose();
  }
}
