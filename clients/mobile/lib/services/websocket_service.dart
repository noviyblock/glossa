import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class GlossItem {
  final String gloss;
  final double prob;
  const GlossItem({required this.gloss, required this.prob});

  factory GlossItem.fromJson(Map<String, dynamic> j) => GlossItem(
        gloss: j['gloss'] as String,
        prob: (j['prob'] as num).toDouble(),
      );
}

enum WsStatus { disconnected, connecting, connected, error }

class WebSocketService extends ChangeNotifier {
  WebSocketChannel? _channel;
  StreamSubscription? _sub;

  WsStatus _status = WsStatus.disconnected;
  WsStatus get status => _status;

  String _statusMessage = '';
  String get statusMessage => _statusMessage;

  // Streams exposed to screens
  final _glossController  = StreamController<List<GlossItem>>.broadcast();
  final _chunkController  = StreamController<String>.broadcast();
  final _resultController = StreamController<Map<String, dynamic>>.broadcast();
  final _audioController  = StreamController<String>.broadcast(); // base64 WAV
  final _errorController  = StreamController<String>.broadcast();

  Stream<List<GlossItem>>        get onGloss  => _glossController.stream;
  Stream<String>                 get onChunk  => _chunkController.stream;
  Stream<Map<String, dynamic>>   get onResult => _resultController.stream;
  Stream<String>                 get onAudio  => _audioController.stream;
  Stream<String>                 get onError  => _errorController.stream;

  // ── Connection ─────────────────────────────────────────────────────────── //

  Future<void> connect(String url, {int maxRetries = 3}) async {
    _setStatus(WsStatus.connecting, 'Подключение…');
    for (int attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        _channel = WebSocketChannel.connect(Uri.parse(url));
        await _channel!.ready;
        _setStatus(WsStatus.connected, 'Подключено');
        _listen();
        return;
      } catch (e) {
        if (attempt == maxRetries) {
          _setStatus(WsStatus.error, 'Ошибка подключения: $e');
          rethrow;
        }
        await Future<void>.delayed(Duration(seconds: attempt * 2));
      }
    }
  }

  void disconnect() {
    _sub?.cancel();
    _channel?.sink.close();
    _channel = null;
    _setStatus(WsStatus.disconnected, '');
  }

  // ── Sending ────────────────────────────────────────────────────────────── //

  void sendVideoFrame(String base64Frame, String sessionId) {
    _send({
      'type': 'video_frame',
      'frame': base64Frame,
      'session_id': sessionId,
    });
  }

  void sendAudioChunk(String base64Audio, String sessionId) {
    _send({
      'type': 'audio_chunk',
      'audio': base64Audio,
      'session_id': sessionId,
    });
  }

  void endSession(String sessionId) {
    _send({'type': 'end_session', 'session_id': sessionId});
  }

  void _send(Map<String, dynamic> msg) {
    if (_channel == null || _status != WsStatus.connected) return;
    _channel!.sink.add(jsonEncode(msg));
  }

  // ── Receiving ──────────────────────────────────────────────────────────── //

  void _listen() {
    _sub = _channel!.stream.listen(
      (raw) {
        try {
          final msg = jsonDecode(raw as String) as Map<String, dynamic>;
          _handleMessage(msg);
        } catch (e) {
          debugPrint('WS parse error: $e');
        }
      },
      onError: (dynamic err) {
        _setStatus(WsStatus.error, 'Ошибка WS: $err');
        _errorController.add('$err');
      },
      onDone: () {
        _setStatus(WsStatus.disconnected, 'Соединение закрыто');
      },
    );
  }

  void _handleMessage(Map<String, dynamic> msg) {
    final type    = msg['type'] as String? ?? '';
    final payload = msg['payload'] as Map<String, dynamic>? ?? {};

    switch (type) {
      case 'gloss':
        final rawList = payload['glosses'] as List<dynamic>? ?? [];
        final glosses = rawList
            .map((e) => GlossItem.fromJson(e as Map<String, dynamic>))
            .toList();
        _glossController.add(glosses);
        break;

      case 'chunk':
        final text = payload['text'] as String? ?? '';
        _chunkController.add(text);
        break;

      case 'result':
        _resultController.add(payload);
        break;

      case 'audio':
        final wav = payload['wav'] as String? ?? '';
        if (wav.isNotEmpty) _audioController.add(wav);
        break;

      case 'error':
        final message = payload['message'] as String? ?? 'Unknown error';
        _errorController.add(message);
        break;
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────── //

  void _setStatus(WsStatus s, String msg) {
    _status = s;
    _statusMessage = msg;
    notifyListeners();
  }

  @override
  void dispose() {
    disconnect();
    _glossController.close();
    _chunkController.close();
    _resultController.close();
    _audioController.close();
    _errorController.close();
    super.dispose();
  }
}
