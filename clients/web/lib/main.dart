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
      ),
      home: const HomePage(),
    );
  }
}
