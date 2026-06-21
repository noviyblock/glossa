/// Central configuration for the Glossa mobile client.
/// Change serverUrl via SettingsScreen at runtime.
class Config {
  static String serverUrl = 'ws://192.168.1.100:8000';

  static String get wsRslToText =>
      '$serverUrl/api/v1/ws/translate/rsl_to_text';

  static String get wsTextToRsl =>
      '$serverUrl/api/v1/ws/translate/text_to_rsl';

  static String get httpBase =>
      serverUrl.replaceFirst('ws://', 'http://').replaceFirst('wss://', 'https://');

  static String get healthUrl => '${httpBase.replaceFirst('ws', 'http')}/health/live';
}
