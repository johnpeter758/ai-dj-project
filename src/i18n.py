#!/usr/bin/env python3
"""
Internationalization (i18n) System for AI DJ Project
Supports multiple languages, locale detection, and translation management.
"""

import os
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import gettext
import locale as locale_module

# Project root directory
PROJECT_ROOT = Path("/Users/johnpeter/ai-dj-project")
SRC_DIR = PROJECT_ROOT / "src"
LOCALES_DIR = SRC_DIR / "locales"

# Default locale settings
DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES = ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "ko", "ru"]

# Current locale (module-level)
_current_locale: str = DEFAULT_LOCALE
_translations: Dict[str, Any] = {}


@dataclass
class LocaleConfig:
    """Locale settings"""
    default_locale: str = DEFAULT_LOCALE
    supported_locales: List[str] = field(default_factory=lambda: SUPPORTED_LOCALES)
    locale_dir: str = str(LOCALES_DIR)
    detect_system_locale: bool = True
    fallback_to_default: bool = True


def get_system_locale() -> str:
    """Detect system locale"""
    try:
        # Try to get system locale
        system_locale = locale_module.getdefaultlocale()
        if system_locale and system_locale[0]:
            lang = system_locale[0].split("_")[0]
            if lang in SUPPORTED_LOCALES:
                return lang
    except (ValueError, AttributeError):
        pass
    
    # Fallback to environment variables
    for env_var in ["LANGUAGE", "LC_ALL", "LANG"]:
        lang = os.getenv(env_var, "")
        if lang:
            lang = lang.split("_")[0]
            if lang in SUPPORTED_LOCALES:
                return lang
    
    return DEFAULT_LOCALE


def set_locale(loc: str) -> bool:
    """Set the current locale"""
    global _current_locale
    
    if loc not in SUPPORTED_LOCALES:
        return False
    
    _current_locale = loc
    return True


def get_locale() -> str:
    """Get the current locale"""
    return _current_locale


def translate(key: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Translate a key to the current locale.
    
    Args:
        key: Translation key (e.g., "ui.play_button")
        default: Default text if key not found
        **kwargs: Format arguments for string interpolation
        
    Returns:
        Translated string
    """
    global _translations
    
    # Try to get translation
    text = _translations.get(_current_locale, {}).get(key)
    
    if text is None and _current_locale != DEFAULT_LOCALE:
        # Fallback to default locale
        text = _translations.get(DEFAULT_LOCALE, {}).get(key)
    
    if text is None:
        # Use default or key itself
        text = default if default else key
    
    # Format if kwargs provided
    if kwargs and isinstance(text, str):
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    
    return text


def translate_plural(key: str, key_plural: str, n: int, **kwargs) -> str:
    """
    Translate a key with pluralization support.
    
    Args:
        key: Translation key for singular
        key_plural: Translation key for plural
        n: Count to determine singular/plural
        **kwargs: Format arguments
        
    Returns:
        Translated string (singular or plural based on n)
    """
    if n == 1:
        return translate(key, **kwargs)
    else:
        return translate(key_plural, **kwargs)


# Translation dictionaries
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # UI Strings
        "ui.play": "Play",
        "ui.pause": "Pause",
        "ui.stop": "Stop",
        "ui.record": "Record",
        "ui.export": "Export",
        "ui.settings": "Settings",
        "ui.load": "Load",
        "ui.save": "Save",
        "ui.cancel": "Cancel",
        "ui.confirm": "Confirm",
        "ui.delete": "Delete",
        "ui.edit": "Edit",
        "ui.add": "Add",
        "ui.remove": "Remove",
        "ui.search": "Search",
        "ui.filter": "Filter",
        "ui.sort": "Sort",
        "ui.refresh": "Refresh",
        "ui.close": "Close",
        "ui.open": "Open",
        "ui.create": "Create",
        "ui.update": "Update",
        
        # Player
        "player.title": "AI DJ Player",
        "player.now_playing": "Now Playing",
        "player.queue": "Queue",
        "player.playlist": "Playlist",
        "player.history": "History",
        "player.shuffle": "Shuffle",
        "player.repeat": "Repeat",
        "player.volume": "Volume",
        "player.bpm": "BPM",
        "player.key": "Key",
        
        # Audio
        "audio.processing": "Processing audio...",
        "audio.analyzing": "Analyzing track...",
        "audio.loading": "Loading audio...",
        "audio.exporting": "Exporting...",
        "audio.mastering": "Mastering...",
        "audio.stem_separation": "Separating stems...",
        "audio.beat_detection": "Detecting beats...",
        
        # Generation
        "gen.creating": "Creating {style}...",
        "gen.melody": "Melody",
        "gen.bass": "Bass",
        "gen.drums": "Drums",
        "gen.vocals": "Vocals",
        "gen.arrangement": "Arrangement",
        "gen.chorus": "Chorus",
        "gen.drop": "Drop",
        "gen.intro": "Intro",
        "gen.outro": "Outro",
        
        # Effects
        "fx.reverb": "Reverb",
        "fx.delay": "Delay",
        "fx.compressor": "Compressor",
        "fx.eq": "Equalizer",
        "fx.filter": "Filter",
        "fx.distortion": "Distortion",
        "fx.chorus": "Chorus",
        "fx.phaser": "Phaser",
        "fx.flanger": "Flanger",
        "fx.mixer": "Mixer",
        "fx.gain": "Gain",
        "fx.mute": "Mute",
        "fx.solo": "Solo",
        
        # Genres
        "genre.electronic": "Electronic",
        "genre.hip_hop": "Hip Hop",
        "genre.pop": "Pop",
        "genre.rock": "Rock",
        "genre.jazz": "Jazz",
        "genre.classical": "Classical",
        "genre.ambient": "Ambient",
        "genre.house": "House",
        "genre.techno": "Techno",
        "genre.trap": "Trap",
        
        # Moods
        "mood.energetic": "Energetic",
        "mood.chill": "Chill",
        "mood.happy": "Happy",
        "mood.sad": "Sad",
        "mood.angry": "Angry",
        "mood.peaceful": "Peaceful",
        "mood.dark": "Dark",
        "mood.uplifting": "Uplifting",
        
        # Status messages
        "status.ready": "Ready",
        "status.playing": "Playing",
        "status.paused": "Paused",
        "status.stopped": "Stopped",
        "status.loading": "Loading...",
        "status.error": "Error",
        "status.success": "Success",
        "status.warning": "Warning",
        "status.complete": "Complete",
        
        # Error messages
        "error.generic": "An error occurred",
        "error.file_not_found": "File not found",
        "error.invalid_format": "Invalid file format",
        "error.audio_device": "Audio device error",
        "error.model_load": "Failed to load model",
        "error.network": "Network error",
        "error.permission": "Permission denied",
        "error.no_audio": "No audio file loaded",
        
        # Tooltips and hints
        "hint.play": "Start playback",
        "hint.pause": "Pause playback",
        "hint.stop": "Stop and reset",
        "hint.record": "Start recording",
        "hint.export": "Export audio file",
        "hint.settings": "Open settings",
        "hint.undo": "Undo last action",
        "hint.redo": "Redo action",
        
        # Dialog titles
        "dialog.confirm_title": "Confirm",
        "dialog.error_title": "Error",
        "dialog.warning_title": "Warning",
        "dialog.info_title": "Information",
        "dialog.save_title": "Save File",
        "dialog.open_title": "Open File",
        
        # Notifications
        "notify.track_added": "Track added to queue",
        "notify.track_removed": "Track removed",
        "notify.export_complete": "Export complete",
        "notify.save_complete": "Save complete",
        "notify.generation_complete": "Generation complete",
        "notify.error": "An error occurred",
    },
    
    "es": {
        # UI Strings
        "ui.play": "Reproducir",
        "ui.pause": "Pausar",
        "ui.stop": "Detener",
        "ui.record": "Grabar",
        "ui.export": "Exportar",
        "ui.settings": "Configuración",
        "ui.load": "Cargar",
        "ui.save": "Guardar",
        "ui.cancel": "Cancelar",
        "ui.confirm": "Confirmar",
        "ui.delete": "Eliminar",
        "ui.edit": "Editar",
        "ui.add": "Añadir",
        "ui.remove": "Quitar",
        "ui.search": "Buscar",
        "ui.filter": "Filtrar",
        "ui.sort": "Ordenar",
        "ui.refresh": "Actualizar",
        "ui.close": "Cerrar",
        "ui.open": "Abrir",
        "ui.create": "Crear",
        "ui.update": "Actualizar",
        
        # Player
        "player.title": "Reproductor AI DJ",
        "player.now_playing": "Reproduciendo",
        "player.queue": "Cola",
        "player.playlist": "Lista",
        "player.history": "Historial",
        "player.shuffle": "Aleatorio",
        "player.repeat": "Repetir",
        "player.volume": "Volumen",
        "player.bpm": "BPM",
        "player.key": "Tonalidad",
        
        # Audio
        "audio.processing": "Procesando audio...",
        "audio.analyzing": "Analizando pista...",
        "audio.loading": "Cargando audio...",
        "audio.exporting": "Exportando...",
        "audio.mastering": "Masterizando...",
        "audio.stem_separation": "Separando stems...",
        "audio.beat_detection": "Detectando beats...",
        
        # Status messages
        "status.ready": "Listo",
        "status.playing": "Reproduciendo",
        "status.paused": "Pausado",
        "status.stopped": "Detenido",
        "status.loading": "Cargando...",
        "status.error": "Error",
        "status.success": "Éxito",
        "status.warning": "Advertencia",
        "status.complete": "Completado",
        
        # Error messages
        "error.generic": "Ocurrió un error",
        "error.file_not_found": "Archivo no encontrado",
        "error.invalid_format": "Formato de archivo inválido",
        "error.audio_device": "Error de dispositivo de audio",
        "error.model_load": "Error al cargar el modelo",
        "error.network": "Error de red",
        "error.permission": "Permiso denegado",
        
        # Genres
        "genre.electronic": "Electrónica",
        "genre.hip_hop": "Hip Hop",
        "genre.pop": "Pop",
        
        # Moods
        "mood.energetic": "Energético",
        "mood.chill": "Relajado",
        "mood.happy": "Feliz",
        "mood.sad": "Triste",
    },
    
    "fr": {
        # UI Strings
        "ui.play": "Lecture",
        "ui.pause": "Pause",
        "ui.stop": "Arrêter",
        "ui.record": "Enregistrer",
        "ui.export": "Exporter",
        "ui.settings": "Paramètres",
        "ui.load": "Charger",
        "ui.save": "Sauvegarder",
        "ui.cancel": "Annuler",
        "ui.confirm": "Confirmer",
        "ui.delete": "Supprimer",
        "ui.edit": "Modifier",
        "ui.add": "Ajouter",
        "ui.remove": "Retirer",
        "ui.search": "Rechercher",
        "ui.filter": "Filtrer",
        "ui.sort": "Trier",
        "ui.refresh": "Actualiser",
        "ui.close": "Fermer",
        "ui.open": "Ouvrir",
        "ui.create": "Créer",
        "ui.update": "Mettre à jour",
        
        # Player
        "player.title": "Lecteur AI DJ",
        "player.now_playing": "En cours",
        "player.queue": "File d'attente",
        "player.playlist": "Playlist",
        "player.history": "Historique",
        "player.shuffle": "Aléatoire",
        "player.repeat": "Répéter",
        "player.volume": "Volume",
        "player.bpm": "BPM",
        "player.key": "Tonalité",
        
        # Audio
        "audio.processing": "Traitement audio...",
        "audio.analyzing": "Analyse de la piste...",
        "audio.loading": "Chargement audio...",
        "audio.exporting": "Exportation...",
        "audio.mastering": "Masterisation...",
        "audio.stem_separation": "Séparation des stems...",
        "audio.beat_detection": "Détection des beats...",
        
        # Status messages
        "status.ready": "Prêt",
        "status.playing": "Lecture",
        "status.paused": "En pause",
        "status.stopped": "Arrêté",
        "status.loading": "Chargement...",
        "status.error": "Erreur",
        "status.success": "Succès",
        "status.warning": "Avertissement",
        "status.complete": "Terminé",
        
        # Error messages
        "error.generic": "Une erreur est survenue",
        "error.file_not_found": "Fichier non trouvé",
        "error.invalid_format": "Format de fichier invalide",
        "error.audio_device": "Erreur du périphérique audio",
        "error.model_load": "Échec du chargement du modèle",
        "error.network": "Erreur réseau",
        "error.permission": "Permission refusée",
        
        # Genres
        "genre.electronic": "Électronique",
        "genre.hip_hop": "Hip Hop",
        "genre.pop": "Pop",
        
        # Moods
        "mood.energetic": "Énergique",
        "mood.chill": "Détendu",
        "mood.happy": "Joyeux",
        "mood.sad": "Triste",
    },
    
    "de": {
        # UI Strings
        "ui.play": "Abspielen",
        "ui.pause": "Pause",
        "ui.stop": "Stopp",
        "ui.record": "Aufnehmen",
        "ui.export": "Exportieren",
        "ui.settings": "Einstellungen",
        "ui.load": "Laden",
        "ui.save": "Speichern",
        "ui.cancel": "Abbrechen",
        "ui.confirm": "Bestätigen",
        "ui.delete": "Löschen",
        "ui.edit": "Bearbeiten",
        "ui.add": "Hinzufügen",
        "ui.remove": "Entfernen",
        "ui.search": "Suchen",
        "ui.filter": "Filtern",
        "ui.sort": "Sortieren",
        "ui.refresh": "Aktualisieren",
        "ui.close": "Schließen",
        "ui.open": "Öffnen",
        "ui.create": "Erstellen",
        "ui.update": "Aktualisieren",
        
        # Player
        "player.title": "AI DJ Player",
        "player.now_playing": "Wird abgespielt",
        "player.queue": "Warteschlange",
        "player.playlist": "Playlist",
        "player.history": "Verlauf",
        "player.shuffle": "Zufällig",
        "player.repeat": "Wiederholen",
        "player.volume": "Lautstärke",
        "player.bpm": "BPM",
        "player.key": "Tonart",
        
        # Audio
        "audio.processing": "Audio wird verarbeitet...",
        "audio.analyzing": "Track wird analysiert...",
        "audio.loading": "Audio wird geladen...",
        "audio.exporting": "Exportieren...",
        "audio.mastering": "Mastering...",
        "audio.stem_separation": "Stems werden getrennt...",
        "audio.beat_detection": "Beats werden erkannt...",
        
        # Status messages
        "status.ready": "Bereit",
        "status.playing": "Wird abgespielt",
        "status.paused": "Pausiert",
        "status.stopped": "Gestoppt",
        "status.loading": "Wird geladen...",
        "status.error": "Fehler",
        "status.success": "Erfolg",
        "status.warning": "Warnung",
        "status.complete": "Abgeschlossen",
        
        # Error messages
        "error.generic": "Ein Fehler ist aufgetreten",
        "error.file_not_found": "Datei nicht gefunden",
        "error.invalid_format": "Ungültiges Dateiformat",
        "error.audio_device": "Audiogerätefehler",
        "error.model_load": "Modell konnte nicht geladen werden",
        "error.network": "Netzwerkfehler",
        "error.permission": "Zugriff verweigert",
        
        # Genres
        "genre.electronic": "Elektronisch",
        "genre.hip_hop": "Hip Hop",
        "genre.pop": "Pop",
        
        # Moods
        "mood.energetic": "Energisch",
        "mood.chill": "Entspannt",
        "mood.happy": "Glücklich",
        "mood.sad": "Traurig",
    },
    
    "ja": {
        # UI Strings
        "ui.play": "再生",
        "ui.pause": "一時停止",
        "ui.stop": "停止",
        "ui.record": "録音",
        "ui.export": "エクスポート",
        "ui.settings": "設定",
        "ui.load": "読み込み",
        "ui.save": "保存",
        "ui.cancel": "キャンセル",
        "ui.confirm": "確認",
        "ui.delete": "削除",
        "ui.edit": "編集",
        "ui.add": "追加",
        "ui.remove": "削除",
        "ui.search": "検索",
        "ui.filter": "フィルター",
        "ui.sort": "並べ替え",
        "ui.refresh": "更新",
        "ui.close": "閉じる",
        "ui.open": "開く",
        "ui.create": "作成",
        "ui.update": "更新",
        
        # Player
        "player.title": "AI DJプレーヤー",
        "player.now_playing": "再生中",
        "player.queue": "キュー",
        "player.playlist": "プレイリスト",
        "player.history": "履歴",
        "player.shuffle": "シャッフル",
        "player.repeat": "リピート",
        "player.volume": "音量",
        "player.bpm": "BPM",
        "player.key": "キー",
        
        # Audio
        "audio.processing": "オーディオ処理中...",
        "audio.analyzing": "トラック分析中...",
        "audio.loading": "オーディオ読み込み中...",
        "audio.exporting": "エクスポート中...",
        "audio.mastering": "マスタリング中...",
        "audio.stem_separation": "ステム分離中...",
        "audio.beat_detection": "ビート検出中...",
        
        # Status messages
        "status.ready": "準備完了",
        "status.playing": "再生中",
        "status.paused": "一時停止中",
        "status.stopped": "停止中",
        "status.loading": "読み込み中...",
        "status.error": "エラー",
        "status.success": "成功",
        "status.warning": "警告",
        "status.complete": "完了",
        
        # Error messages
        "error.generic": "エラーが発生しました",
        "error.file_not_found": "ファイルが見つかりません",
        "error.invalid_format": "無効なファイル形式",
        "error.audio_device": "オーディオデバイスエラー",
        "error.model_load": "モデルの読み込みに失敗しました",
        "error.network": "ネットワークエラー",
        "error.permission": "権限がありません",
        
        # Genres
        "genre.electronic": "エレクトロニック",
        "genre.hip_hop": "ヒップホップ",
        "genre.pop": "ポップ",
        
        # Moods
        "mood.energetic": "エネルギッシュ",
        "mood.chill": "チル",
        "mood.happy": "幸せ",
        "mood.sad": "悲しい",
    },
    
    "zh": {
        # UI Strings
        "ui.play": "播放",
        "ui.pause": "暂停",
        "ui.stop": "停止",
        "ui.record": "录音",
        "ui.export": "导出",
        "ui.settings": "设置",
        "ui.load": "加载",
        "ui.save": "保存",
        "ui.cancel": "取消",
        "ui.confirm": "确认",
        "ui.delete": "删除",
        "ui.edit": "编辑",
        "ui.add": "添加",
        "ui.remove": "移除",
        "ui.search": "搜索",
        "ui.filter": "筛选",
        "ui.sort": "排序",
        "ui.refresh": "刷新",
        "ui.close": "关闭",
        "ui.open": "打开",
        "ui.create": "创建",
        "ui.update": "更新",
        
        # Player
        "player.title": "AI DJ播放器",
        "player.now_playing": "正在播放",
        "player.queue": "播放队列",
        "player.playlist": "播放列表",
        "player.history": "历史记录",
        "player.shuffle": "随机播放",
        "player.repeat": "重复",
        "player.volume": "音量",
        "player.bpm": "BPM",
        "player.key": "调",
        
        # Audio
        "audio.processing": "处理音频中...",
        "audio.analyzing": "分析音轨中...",
        "audio.loading": "加载音频中...",
        "audio.exporting": "导出中...",
        "audio.mastering": "母带处理中...",
        "audio.stem_separation": "分离音轨中...",
        "audio.beat_detection": "检测节拍中...",
        
        # Status messages
        "status.ready": "就绪",
        "status.playing": "播放中",
        "status.paused": "已暂停",
        "status.stopped": "已停止",
        "status.loading": "加载中...",
        "status.error": "错误",
        "status.success": "成功",
        "status.warning": "警告",
        "status.complete": "完成",
        
        # Error messages
        "error.generic": "发生错误",
        "error.file_not_found": "文件未找到",
        "error.invalid_format": "无效的文件格式",
        "error.audio_device": "音频设备错误",
        "error.model_load": "模型加载失败",
        "error.network": "网络错误",
        "error.permission": "权限被拒绝",
        
        # Genres
        "genre.electronic": "电子",
        "genre.hip_hop": "嘻哈",
        "genre.pop": "流行",
        
        # Moods
        "mood.energetic": "充满活力",
        "mood.chill": "放松",
        "mood.happy": "快乐",
        "mood.sad": "悲伤",
    },
}


def init_i18n(config: Optional[LocaleConfig] = None) -> None:
    """Initialize the i18n system.
    
    Args:
        config: Optional LocaleConfig to customize behavior
    """
    global _translations, _current_locale
    
    # Load all translations
    _translations = TRANSLATIONS.copy()
    
    # Try to load custom translations from locale files
    _load_custom_translations()
    
    # Detect and set locale
    if config and not config.detect_system_locale:
        _current_locale = config.default_locale
    else:
        _current_locale = get_system_locale()
    
    # Validate locale
    if _current_locale not in SUPPORTED_LOCALES:
        _current_locale = DEFAULT_LOCALE


def _load_custom_translations() -> None:
    """Load custom translations from locale files if they exist"""
    locale_dir = LOCALES_DIR
    
    if not locale_dir.exists():
        # Create locales directory
        locale_dir.mkdir(parents=True, exist_ok=True)
        return
    
    # Look for JSON translation files
    for locale in SUPPORTED_LOCALES:
        locale_file = locale_dir / f"{locale}.json"
        if locale_file.exists():
            try:
                with open(locale_file, "r", encoding="utf-8") as f:
                    custom_trans = json.load(f)
                    if locale in _translations:
                        _translations[locale].update(custom_trans)
                    else:
                        _translations[locale] = custom_trans
            except (json.JSONDecodeError, IOError):
                pass


def get_available_translations() -> Dict[str, int]:
    """
    Get count of available translations per locale.
    
    Returns:
        Dictionary mapping locale to translation count
    """
    return {
        locale: len(translations) 
        for locale, translations in _translations.items()
    }


def add_translation(locale: str, key: str, value: str) -> None:
    """
    Add or update a translation at runtime.
    
    Args:
        locale: Locale code (e.g., "en", "es")
        key: Translation key
        value: Translation value
    """
    global _translations
    
    if locale not in _translations:
        _translations[locale] = {}
    
    _translations[locale][key] = value


def get_supported_locales() -> List[str]:
    """Get list of supported locale codes"""
    return SUPPORTED_LOCALES.copy()


def get_locale_name(locale: str) -> str:
    """Get human-readable locale name"""
    locale_names = {
        "en": "English",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
        "it": "Italiano",
        "pt": "Português",
        "ja": "日本語",
        "zh": "中文",
        "ko": "한국어",
        "ru": "Русский",
    }
    return locale_names.get(locale, locale)


# Convenience functions for common translations
def t(key: str, default: Optional[str] = None, **kwargs) -> str:
    """Short form for translate()"""
    return translate(key, default, **kwargs)


def tp(key: str, key_plural: str, n: int, **kwargs) -> str:
    """Short form for translate_plural()"""
    return translate_plural(key, key_plural, n, **kwargs)


# Initialize on import
init_i18n()


# Export public API
__all__ = [
    "LocaleConfig",
    "set_locale",
    "get_locale",
    "translate",
    "translate_plural",
    "t",
    "tp",
    "init_i18n",
    "get_system_locale",
    "get_available_translations",
    "add_translation",
    "get_supported_locales",
    "get_locale_name",
    "SUPPORTED_LOCALES",
    "DEFAULT_LOCALE",
    "LOCALES_DIR",
]
