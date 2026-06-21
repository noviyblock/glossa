import 'package:flutter/material.dart';

import '../services/websocket_service.dart';
import 'confidence_bar.dart';

class GlossCard extends StatelessWidget {
  final GlossItem item;
  final int rank; // 1 = top-1

  const GlossCard({super.key, required this.item, required this.rank});

  @override
  Widget build(BuildContext context) {
    final cs  = Theme.of(context).colorScheme;
    final pct = (item.prob * 100).toStringAsFixed(1);

    return Card(
      elevation: rank == 1 ? 4 : 1,
      color: rank == 1 ? cs.primaryContainer : cs.surfaceContainerLow,
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            // Rank badge
            CircleAvatar(
              radius: 14,
              backgroundColor: rank == 1 ? cs.primary : cs.outlineVariant,
              child: Text(
                '$rank',
                style: TextStyle(
                  color: rank == 1 ? cs.onPrimary : cs.onSurface,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(width: 12),
            // Gloss label
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.gloss,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: rank == 1 ? FontWeight.bold : FontWeight.normal,
                        ),
                  ),
                  const SizedBox(height: 4),
                  ConfidenceBar(value: item.prob),
                ],
              ),
            ),
            const SizedBox(width: 12),
            // Percentage
            Text(
              '$pct%',
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: cs.onSurfaceVariant,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}
