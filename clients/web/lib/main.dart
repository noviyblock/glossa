import 'package:flutter/material.dart';

import 'pages/home_page.dart';

void main() {
  runApp(const GlossaWebApp());
}

class GlossaWebApp extends StatelessWidget {
  const GlossaWebApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Glossa Web',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.deepPurple,
        // Explicit, not left to the Material default -- without this,
        // CanvasKit falls back to fetching Roboto from fonts.gstatic.com at
        // runtime instead of using the bundled font declared in
        // pubspec.yaml (assets/fonts/), and on a network that can't reach
        // it every bit of text in the app silently renders blank.
        fontFamily: 'Roboto',
      ),
      home: const HomePage(),
    );
  }
}
