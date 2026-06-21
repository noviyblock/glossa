/// Central configuration for the Glossa web client.
class Config {
  // Override via URL parameter ?server=ws://host:8000 or env at build time.
  static String serverUrl =
      const String.fromEnvironment('SERVER_URL', defaultValue: 'ws://localhost:8000');

  static String get wsRslToText =>
      '$serverUrl/api/v1/ws/translate/rsl_to_text';

  static String get wsTextToRsl =>
      '$serverUrl/api/v1/ws/translate/text_to_rsl';

  static String get httpBase =>
      serverUrl.replaceFirst('ws://', 'http://').replaceFirst('wss://', 'https://');
}
