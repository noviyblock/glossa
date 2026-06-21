import 'package:flutter/material.dart';

class ConfidenceBar extends StatelessWidget {
  final double value; // 0.0 – 1.0
  final Color? color;
  final double height;

  const ConfidenceBar({
    super.key,
    required this.value,
    this.color,
    this.height = 6,
  });

  @override
  Widget build(BuildContext context) {
    final cs    = Theme.of(context).colorScheme;
    final clamp = value.clamp(0.0, 1.0).toDouble();
    final barColor = color ??
        (clamp >= 0.7
            ? cs.primary
            : clamp >= 0.4
                ? cs.secondary
                : cs.error);

    return ClipRRect(
      borderRadius: BorderRadius.circular(height / 2),
      child: LinearProgressIndicator(
        value: clamp,
        minHeight: height,
        backgroundColor: cs.surfaceContainerHighest,
        valueColor: AlwaysStoppedAnimation<Color>(barColor),
      ),
    );
  }
}
