"""Custom exceptions for the AI DJ Project."""


class AIDJException(Exception):
    """Base exception for all AI DJ errors."""
    pass


# File/IO Exceptions
class FileNotFoundException(AIDJException):
    """Raised when an audio file cannot be found."""
    pass


class InvalidFileFormatException(AIDJException):
    """Raised when a file format is not supported."""
    pass


class CorruptedFileException(AIDJException):
    """Raised when an audio file is corrupted or unreadable."""
    pass


# Analysis Exceptions
class AnalysisException(AIDJException):
    """Base exception for analysis-related errors."""
    pass


class AnalysisTimeoutException(AnalysisException):
    """Raised when audio analysis takes too long."""
    pass


class InsufficientAudioDataException(AnalysisException):
    """Raised when there's not enough audio data to analyze."""
    pass


# Playback Exceptions
class PlaybackException(AIDJException):
    """Base exception for playback-related errors."""
    pass


class NoAudioDeviceException(PlaybackException):
    """Raised when no audio output device is available."""
    pass


class PlaybackRateException(PlaybackException):
    """Raised when playback rate is invalid or unsupported."""
    pass


# Configuration Exceptions
class ConfigurationException(AIDJException):
    """Raised when there's a configuration error."""
    pass


class MissingConfigurationException(ConfigurationException):
    """Raised when a required configuration value is missing."""
    pass


# API Exceptions
class APIException(AIDJException):
    """Base exception for external API errors."""
    pass


class APIRateLimitException(APIException):
    """Raised when API rate limit is exceeded."""
    pass


class APIAuthenticationException(APIException):
    """Raised when API authentication fails."""
    pass
