import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

class CameraService {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isStreaming = false;
  bool _capturing   = false;
  Timer? _timer;

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
    _isStreaming = true;
    _timer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
      if (_capturing) return;
      _capturing = true;
      try {
        final frame = await _captureFrame();
        if (frame != null) onFrame(frame);
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

  Future<String?> _captureFrame() async {
    if (_controller == null || !_controller!.value.isInitialized) return null;
    final file = await _controller!.takePicture();
    final bytes = await file.readAsBytes();
    // Clean up temp file
    try { await File(file.path).delete(); } catch (_) {}
    return base64Encode(bytes);
  }

  // ── Cleanup ────────────────────────────────────────────────────────────── //

  Future<void> dispose() async {
    stopStreaming();
    await _controller?.dispose();
    _controller = null;
  }
}
