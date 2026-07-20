import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

class CameraService {
  // Motion gating: skip sending frames to the backend while the scene is
  // static, so the (expensive) DWPose+ST-GCN pipeline only runs once someone
  // actually starts gesturing. A 1s heartbeat keeps the session alive even
  // if the threshold misses very slow motion.
  static const int _motionGridSize  = 16;
  static const int _motionThreshold = 6; // mean abs luminance diff, 0-255 scale
  static const Duration _heartbeatInterval = Duration(seconds: 1);

  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isStreaming = false;
  bool _capturing   = false;
  Timer? _timer;

  Uint8List?  _lastGray;
  DateTime    _lastSentAt = DateTime.fromMillisecondsSinceEpoch(0);

  bool get isInitialized => _controller?.value.isInitialized ?? false;
  bool get isStreaming    => _isStreaming;
  CameraController? get controller => _controller;

  // ── Initialization ─────────────────────────────────────────────────────── //

  Future<void> initialize({int cameraIndex = 0}) async {
    _cameras = await availableCameras();
    if (_cameras.isEmpty) throw Exception('No cameras found');

    final desc = _cameras[cameraIndex.clamp(0, _cameras.length - 1)];
    _controller = CameraController(
      desc,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await _controller!.initialize();
  }

  Future<void> switchCamera() async {
    if (_cameras.length < 2) return;
    final currentIdx = _cameras.indexOf(_controller!.description);
    await _controller?.dispose();
    await initialize(cameraIndex: (currentIdx + 1) % _cameras.length);
  }

  // ── Frame streaming (10 fps = 100ms interval) ──────────────────────────── //

  void startStreaming(void Function(String base64Jpeg) onFrame) {
    if (!isInitialized || _isStreaming) return;
    _isStreaming  = true;
    _lastGray     = null;
    _lastSentAt   = DateTime.fromMillisecondsSinceEpoch(0);
    _timer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
      if (_capturing) return;
      _capturing = true;
      try {
        final bytes = await _captureFrameBytes();
        if (bytes != null && _shouldSend(bytes)) {
          onFrame(base64Encode(bytes));
        }
      } catch (e) {
        debugPrint('Frame capture error: $e');
      } finally {
        _capturing = false;
      }
    });
  }

  void stopStreaming() {
    _timer?.cancel();
    _timer = null;
    _isStreaming = false;
    _capturing   = false;
  }

  Future<Uint8List?> _captureFrameBytes() async {
    if (_controller == null || !_controller!.value.isInitialized) return null;
    final file = await _controller!.takePicture();
    final bytes = await file.readAsBytes();
    // Clean up temp file
    try { await File(file.path).delete(); } catch (_) {}
    return bytes;
  }

  /// Cheap motion check: downsample to a small grayscale grid and compare
  /// mean luminance delta against the previous frame. Always sends on the
  /// first frame and at least once per [_heartbeatInterval] regardless of
  /// motion, so a slow gesture start or analysis failure can't stall a
  /// session indefinitely.
  bool _shouldSend(Uint8List jpegBytes) {
    final now = DateTime.now();
    final heartbeatDue = now.difference(_lastSentAt) >= _heartbeatInterval;

    final decoded = img.decodeJpg(jpegBytes);
    if (decoded == null) {
      _lastSentAt = now;
      return true; // can't analyze — fail open
    }

    final small = img.copyResize(decoded, width: _motionGridSize, height: _motionGridSize);
    final gray = Uint8List(_motionGridSize * _motionGridSize);
    var i = 0;
    for (var y = 0; y < _motionGridSize; y++) {
      for (var x = 0; x < _motionGridSize; x++) {
        final p = small.getPixel(x, y);
        final lum = 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;
        gray[i++] = lum.round().clamp(0, 255);
      }
    }

    bool motion = true;
    final prev = _lastGray;
    if (prev != null) {
      var diff = 0;
      for (var j = 0; j < gray.length; j++) {
        diff += (gray[j] - prev[j]).abs();
      }
      motion = (diff / gray.length) >= _motionThreshold;
    }
    _lastGray = gray;

    if (motion || heartbeatDue) {
      _lastSentAt = now;
      return true;
    }
    return false;
  }

  // ── Cleanup ────────────────────────────────────────────────────────────── //

  Future<void> dispose() async {
    stopStreaming();
    await _controller?.dispose();
    _controller = null;
  }
}
