#!/usr/bin/env python3
"""
API Client Module for AI DJ Project

Provides unified interface for external APIs:
- OpenAI (GPT, Whisper, TTS)
- ElevenLabs (Text-to-Speech)
- Spotify (Music metadata, playlists)
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(APIError):
    """Rate limit exceeded"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Authentication failed"""
    def __init__(self, message: str):
        super().__init__(message, status_code=401)


@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any = None
    error: str = None
    status_code: int = None
    headers: Dict = field(default_factory=dict)


class BaseAPIClient(ABC):
    """Base class for API clients with retry and error handling"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5
    ):
        self.api_key = api_key or self._get_api_key_from_env()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        self.session = self._create_session()
    
    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable"""
        pass
    
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get default headers - override in subclasses"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        json_data: Dict = None,
        data: Any = None,
        headers: Dict = None,
        timeout: int = None
    ) -> APIResponse:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout
        
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                data=data,
                headers=request_headers,
                timeout=timeout
            )
            
            return self._handle_response(response)
            
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error=f"Request timeout after {timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            return APIResponse(
                success=False,
                error=f"Connection error: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            return APIResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )
    
    def _handle_response(self, response: requests.Response) -> APIResponse:
        """Handle API response and errors"""
        status_code = response.status_code
        
        # Try to parse JSON
        try:
            data = response.json()
        except ValueError:
            data = response.text if response.text else None
        
        if status_code == 200:
            return APIResponse(success=True, data=data, status_code=status_code)
        
        # Handle specific errors
        if status_code == 401:
            error_msg = data.get("error", {}).get("message", "Authentication failed") if isinstance(data, dict) else "Authentication failed"
            return APIResponse(success=False, error=error_msg, status_code=status_code)
        
        if status_code == 429:
            retry_after = response.headers.get("Retry-After", 60)
            error_msg = data.get("error", {}).get("message", "Rate limit exceeded") if isinstance(data, dict) else "Rate limit exceeded"
            return APIResponse(success=False, error=error_msg, status_code=status_code)
        
        if status_code >= 500:
            error_msg = data.get("error", {}).get("message", "Server error") if isinstance(data, dict) else "Server error"
            return APIResponse(success=False, error=error_msg, status_code=status_code)
        
        # Client errors (4xx except 401/429)
        error_msg = data.get("error", {}).get("message", str(data)) if isinstance(data, dict) else str(data)
        return APIResponse(success=False, error=error_msg, status_code=status_code)
    
    def get(self, endpoint: str, params: Dict = None) -> APIResponse:
        """GET request"""
        return self._make_request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, json_data: Dict = None, data: Any = None) -> APIResponse:
        """POST request"""
        return self._make_request("POST", endpoint, json_data=json_data, data=data)
    
    def put(self, endpoint: str, json_data: Dict = None) -> APIResponse:
        """PUT request"""
        return self._make_request("PUT", endpoint, json_data=json_data)
    
    def delete(self, endpoint: str) -> APIResponse:
        """DELETE request"""
        return self._make_request("DELETE", endpoint)


class OpenAIClient(BaseAPIClient):
    """OpenAI API client"""
    
    BASE_URL = "https://api.openai.com/v1"
    
    class Model(Enum):
        GPT_4O = "gpt-4o"
        GPT_4O_MINI = "gpt-4o-mini"
        GPT_4_TURBO = "gpt-4-turbo"
        GPT_35_TURBO = "gpt-3.5-turbo"
        WHISPER_1 = "whisper-1"
        TTS_1 = "tts-1"
        TTS_1_HD = "tts-1-hd"
    
    def __init__(self, api_key: Optional[str] = None, organization: str = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)
        self.organization = organization
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")
    
    def _get_headers(self) -> Dict[str, str]:
        headers = super()._get_headers()
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers
    
    # --- Chat Completions ---
    def chat_completion(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        stream: bool = False,
        **kwargs
    ) -> APIResponse:
        """Create chat completion"""
        model = model or self.Model.GPT_4O_MINI.value
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
           
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        payload.update(kwargs)
        
        return self.post("/chat/completions", json_data=payload)
    
    # --- Audio Transcription ---
    def transcription(
        self,
        audio_file: Union[str, Path, bytes],
        model: str = "whisper-1",
        language: str = None,
        prompt: str = None,
        response_format: str = "json"
    ) -> APIResponse:
        """Transcribe audio file"""
        files = {"file": audio_file} if isinstance(audio_file, bytes) else open(audio_file, "rb")
        
        data = {"model": model, "response_format": response_format}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        
        headers = {"Content-Type": None}  # Let requests set multipart
        
        try:
            response = self.session.post(
                f"{self.base_url}/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=120
            )
            return self._handle_response(response)
        finally:
            if isinstance(audio_file, str) or isinstance(audio_file, Path):
                files.close()
    
    # --- Text to Speech ---
    def text_to_speech(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0
    ) -> APIResponse:
        """Generate speech from text"""
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed
        }
        
        headers = {"Accept": "application/octet-stream"}
        
        return self.post("/audio/speech", json_data=payload, headers=headers)
    
    # --- Embeddings ---
    def embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small"
    ) -> APIResponse:
        """Create embeddings"""
        if isinstance(input_text, str):
            input_text = [input_text]
        
        payload = {
            "model": model,
            "input": input_text
        }
        
        return self.post("/embeddings", json_data=payload)


class ElevenLabsClient(BaseAPIClient):
    """ElevenLabs API client"""
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    # Default voices
    VOICES = {
        "adam": "pNInz6obpgDQGcFmaJgB",
        "rachel": "CwhRBWXzGAHq8TQ4Gs17",
        "domi": "AZnzlk1XvdvUeBnObPgC",
        "ella": "MF3mGyEYCl7XYWbV9VgO",
        "fin": "D38z5RcWu1voky8BS1Oj",
        "aria": "JDi7Uv2hTG4mQfJKztWA",
        "callum": "N2lVS1wR3CNcPwXLjrYD"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=self.BASE_URL)
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("ELEVENLABS_API_KEY")
    
    def _get_headers(self) -> Dict[str, str]:
        headers = super()._get_headers()
        headers["Accept"] = "application/json"
        return headers
    
    def get_voices(self) -> APIResponse:
        """Get available voices"""
        return self.get("/voices")
    
    def get_voice_settings(self, voice_id: str) -> APIResponse:
        """Get settings for a specific voice"""
        return self.get(f"/voices/{voice_id}/settings")
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        voice_settings: Dict = None,
        model_id: str = "eleven_monolingual_v1",
        optimize_latency: int = 0
    ) -> APIResponse:
        """Convert text to speech"""
        voice_id = voice_id or self.VOICES["adam"]
        
        payload = {
            "text": text,
            "model_id": model_id,
            "optimize_latency": optimize_latency
        }
        
        if voice_settings:
            payload["voice_settings"] = voice_settings
        else:
            payload["voice_settings"] = {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        
        headers = {"Accept": "audio/mpeg"}
        
        return self.post(f"/text-to-speech/{voice_id}", json_data=payload, headers=headers)
    
    def text_to_speech_streaming(
        self,
        text: str,
        voice_id: str = None,
        voice_settings: Dict = None,
        model_id: str = "eleven_monolingual_v1"
    ) -> requests.Response:
        """Stream text to speech (returns raw response for streaming)"""
        voice_id = voice_id or self.VOICES["adam"]
        
        payload = {
            "text": text,
            "model_id": model_id
        }
        
        if voice_settings:
            payload["voice_settings"] = voice_settings
        
        headers = {"Accept": "audio/mpeg"}
        
        response = self.session.post(
            f"{self.base_url}/text-to-speech/{voice_id}/stream",
            json=payload,
            headers={**self._get_headers(), **headers},
            stream=True,
            timeout=self.timeout
        )
        
        return response
    
    def get_history(self, page_size: int = 30) -> APIResponse:
        """Get synthesis history"""
        return self.get(f"/history?page_size={page_size}")
    
    def get_history_item(self, history_item_id: str) -> APIResponse:
        """Get specific history item"""
        return self.get(f"/history/{history_item_id}")
    
    def delete_history_item(self, history_item_id: str) -> APIResponse:
        """Delete a history item"""
        return self.delete(f"/history/{history_item_id}")


class SpotifyClient(BaseAPIClient):
    """Spotify API client"""
    
    BASE_URL = "https://api.spotify.com/v1"
    AUTH_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ):
        super().__init__(base_url=self.BASE_URL)
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self._access_token = None
        self._token_expiry = 0
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return None  # Uses OAuth flow
    
    def _get_access_token(self) -> str:
        """Get OAuth access token"""
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
        
        if not self.client_id or not self.client_secret:
            raise AuthenticationError("Spotify client credentials not configured")
        
        # Get new token
        auth = (self.client_id, self.client_secret)
        response = requests.post(
            self.AUTH_URL,
            data={"grant_type": "client_credentials"},
            auth=auth,
            timeout=30
        )
        
        if response.status_code != 200:
            raise AuthenticationError(f"Failed to get access token: {response.text}")
        
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data["expires_in"] - 60
        
        return self._access_token
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json"
        }
    
    # --- Search ---
    def search(
        self,
        query: str,
        types: List[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> APIResponse:
        """Search Spotify"""
        types = types or ["track"]
        params = {
            "q": query,
            "type": ",".join(types),
            "limit": limit,
            "offset": offset
        }
        return self.get("/search", params=params)
    
    def search_track(self, query: str, limit: int = 10) -> APIResponse:
        """Search for tracks"""
        return self.search(query, types=["track"], limit=limit)
    
    def search_artist(self, query: str, limit: int = 20) -> APIResponse:
        """Search for artists"""
        return self.search(query, types=["artist"], limit=limit)
    
    # --- Tracks ---
    def get_track(self, track_id: str) -> APIResponse:
        """Get track details"""
        return self.get(f"/tracks/{track_id}")
    
    def get_tracks(self, track_ids: List[str]) -> APIResponse:
        """Get multiple track details"""
        ids = ",".join(track_ids)
        return self.get("/tracks", params={"ids": ids})
    
    def get_audio_features(self, track_id: str) -> APIResponse:
        """Get audio features for a track"""
        return self.get(f"/audio-features/{track_id}")
    
    def get_audio_features_batch(self, track_ids: List[str]) -> APIResponse:
        """Get audio features for multiple tracks"""
        ids = ",".join(track_ids)
        return self.get("/audio-features", params={"ids": ids})
    
    # --- Artists ---
    def get_artist(self, artist_id: str) -> APIResponse:
        """Get artist details"""
        return self.get(f"/artists/{artist_id}")
    
    def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> APIResponse:
        """Get artist's top tracks"""
        return self.get(f"/artists/{artist_id}/top-tracks", params={"market": market})
    
    def get_related_artists(self, artist_id: str) -> APIResponse:
        """Get related artists"""
        return self.get(f"/artists/{artist_id}/related-artists")
    
    # --- Albums ---
    def get_album(self, album_id: str) -> APIResponse:
        """Get album details"""
        return self.get(f"/albums/{album_id}")
    
    def get_album_tracks(self, album_id: str, limit: int = 50, offset: int = 0) -> APIResponse:
        """Get album tracks"""
        return self.get(f"/albums/{album_id}/tracks", params={"limit": limit, "offset": offset})
    
    # --- Playlists ---
    def get_playlist(self, playlist_id: str) -> APIResponse:
        """Get playlist details"""
        return self.get(f"/playlists/{playlist_id}")
    
    def get_playlist_tracks(
        self,
        playlist_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: str = None
    ) -> APIResponse:
        """Get playlist tracks"""
        params = {"limit": limit, "offset": offset}
        if fields:
            params["fields"] = fields
        return self.get(f"/playlists/{playlist_id}/tracks", params=params)
    
    # --- Genres & Recommendations ---
    def get_available_genre_seeds(self) -> APIResponse:
        """Get available genre seeds for recommendations"""
        return self.get("/recommendations/available-genre-seeds")
    
    def get_recommendations(
        self,
        seed_genres: List[str] = None,
        seed_artists: List[str] = None,
        seed_tracks: List[str] = None,
        limit: int = 20,
        market: str = "US",
        **kwargs
    ) -> APIResponse:
        """Get track recommendations"""
        params = {
            "limit": limit,
            "market": market
        }
        
        if seed_genres:
            params["seed_genres"] = ",".join(seed_genres)
        if seed_artists:
            params["seed_artists"] = ",".join(seed_artists)
        if seed_tracks:
            params["seed_tracks"] = ",".join(seed_tracks)
        
        # Add any additional parameters (target_tempo, target_energy, etc.)
        params.update(kwargs)
        
        return self.get("/recommendations", params=params)
    
    # --- Audio Analysis ---
    def get_audio_analysis(self, track_id: str) -> APIResponse:
        """Get detailed audio analysis for a track"""
        return self.get(f"/audio-analysis/{track_id}")
    
    # --- User Profile ---
    def get_current_user(self) -> APIResponse:
        """Get current user profile"""
        return self.get("/me")
    
    def get_user_playlists(self, limit: int = 50) -> APIResponse:
        """Get current user's playlists"""
        return self.get("/me/playlists", params={"limit": limit})


# --- Convenience factory functions ---

def get_openai_client(api_key: str = None) -> OpenAIClient:
    """Get OpenAI client instance"""
    return OpenAIClient(api_key=api_key)


def get_elevenlabs_client(api_key: str = None) -> ElevenLabsClient:
    """Get ElevenLabs client instance"""
    return ElevenLabsClient(api_key=api_key)


def get_spotify_client(
    client_id: str = None,
    client_secret: str = None
) -> SpotifyClient:
    """Get Spotify client instance"""
    return SpotifyClient(client_id=client_id, client_secret=client_secret)


# --- Unified APIClient for easy access ---

class APIClient:
    """Unified API client providing access to all services"""
    
    def __init__(
        self,
        openai_key: str = None,
        elevenlabs_key: str = None,
        spotify_client_id: str = None,
        spotify_client_secret: str = None
    ):
        self.openai = OpenAIClient(api_key=openai_key)
        self.elevenlabs = ElevenLabsClient(api_key=elevenlabs_key)
        self.spotify = SpotifyClient(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret
        )
    
    @classmethod
    def from_env(cls) -> "APIClient":
        """Create client from environment variables"""
        return cls(
            openai_key=os.getenv("OPENAI_API_KEY"),
            elevenlabs_key=os.getenv("ELEVENLABS_API_KEY"),
            spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
        )
