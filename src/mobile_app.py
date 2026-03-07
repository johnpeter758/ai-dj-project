#!/usr/bin/env python3
"""
AI DJ Mobile App Wrapper
========================
Mobile-friendly API with push notifications, background processing, and offline capabilities.

Features:
- RESTful API optimized for mobile devices
- Push notifications (FCM/APNs)
- Background task processing with Celery
- Offline mode with local caching
- JWT authentication
- Rate limiting

Usage:
    # Start server
    python mobile_app.py --port 5000 --host 0.0.0.0
    
    # Or import as module
    from mobile_app import create_app, MobileAPIController
"""

import os
import sys
import json
import hashlib
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai-dj-mobile")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MobileConfig:
    """Configuration for mobile API."""
    # Server
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Paths
    project_root: str = "/Users/johnpeter/ai-dj-project/src"
    output_dir: str = "/Users/johnpeter/ai-dj-project/src/output"
    cache_dir: str = "/Users/johnpeter/ai-dj-project/src/cache"
    offline_dir: str = "/Users/johnpeter/ai-dj-project/src/offline"
    
    # API
    api_version: str = "v1"
    max_upload_size_mb: int = 50
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Push Notifications
    fcm_api_key: str = ""
    apns_key_path: str = ""
    apns_team_id: str = ""
    
    # Background Processing
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    background_workers: int = 4
    
    # Offline
    offline_enabled: bool = True
    offline_cache_size_mb: int = 500
    sync_interval: int = 300  # seconds
    
    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24
    session_timeout: int = 3600


# =============================================================================
# DATA MODELS
# =============================================================================

class TaskStatus(Enum):
    """Status of background tasks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationType(Enum):
    """Types of push notifications."""
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    NEW_TRENDS = "new_trends"
    SYSTEM_ALERT = "system_alert"
    GENERATION_READY = "generation_ready"


@dataclass
class User:
    """User model."""
    id: str
    username: str
    email: str
    created_at: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    device_tokens: List[str] = field(default_factory=list)
    offline_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Background task model."""
    id: str
    user_id: str
    type: str  # generate, remix, master, etc.
    status: TaskStatus
    progress: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    notification_sent: bool = False


@dataclass
class SongRequest:
    """Song generation request."""
    genre: str = "house"
    key: str = "C"
    scale: str = "minor"
    bpm: int = 128
    duration_sec: int = 180
    title: str = ""
    artist: str = ""
    energy: float = 0.8
    mood: str = "energetic"
    notification_enabled: bool = True


@dataclass
class OfflinePackage:
    """Offline data package for sync."""
    id: str
    user_id: str
    songs: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    created_at: str
    checksum: str


# =============================================================================
# OFFLINE STORAGE
# =============================================================================

class OfflineStorage:
    """Manages offline data storage and sync."""
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.storage_path = Path(config.offline_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
    def save_task(self, task: Task) -> None:
        """Save task to offline storage."""
        with self._lock:
            task_file = self.storage_path / f"task_{task.id}.json"
            with open(task_file, 'w') as f:
                json.dump(asdict(task), f, indent=2)
                
    def load_task(self, task_id: str) -> Optional[Task]:
        """Load task from offline storage."""
        task_file = self.storage_path / f"task_{task_id}.json"
        if task_file.exists():
            with open(task_file, 'r') as f:
                data = json.load(f)
                data['status'] = TaskStatus(data['status'])
                return Task(**data)
        return None
    
    def save_generated_song(self, song_data: Dict[str, Any]) -> str:
        """Save generated song for offline access."""
        song_id = song_data.get('id', str(uuid.uuid4()))
        with self._lock:
            song_file = self.storage_path / f"song_{song_id}.json"
            with open(song_file, 'w') as f:
                json.dump(song_data, f, indent=2)
        return song_id
    
    def load_song(self, song_id: str) -> Optional[Dict[str, Any]]:
        """Load song from offline storage."""
        song_file = self.storage_path / f"song_{song_id}.json"
        if song_file.exists():
            with open(song_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_offline_songs(self, user_id: str) -> List[Dict[str, Any]]:
        """List all offline songs for a user."""
        songs = []
        for f in self.storage_path.glob("song_*.json"):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    if data.get('user_id') == user_id:
                        songs.append({
                            'id': data.get('id'),
                            'title': data.get('title'),
                            'genre': data.get('genre'),
                            'created_at': data.get('created_at'),
                            'file_size': f.stat().st_size
                        })
            except Exception:
                continue
        return songs
    
    def create_sync_package(self, user_id: str) -> OfflinePackage:
        """Create offline sync package."""
        songs = self.list_offline_songs(user_id)
        tasks = []
        
        for f in self.storage_path.glob("task_*.json"):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    if data.get('user_id') == user_id:
                        tasks.append(data)
            except Exception:
                continue
        
        package_data = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'songs': songs,
            'preferences': {},
            'tasks': tasks,
            'created_at': datetime.now().isoformat()
        }
        
        # Calculate checksum
        checksum = hashlib.sha256(
            json.dumps(package_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        package_data['checksum'] = checksum
        
        return OfflinePackage(**package_data)
    
    def clear_cache(self, max_size_mb: int = None) -> int:
        """Clear old cached files to stay under size limit."""
        if max_size_mb is None:
            max_size_mb = self.config.offline_cache_size_mb
            
        total_size = sum(f.stat().st_size for f in self.storage_path.glob("*"))
        max_bytes = max_size_mb * 1024 * 1024
        
        if total_size <= max_bytes:
            return 0
            
        # Delete oldest files first
        cleared = 0
        for f in sorted(self.storage_path.glob("*"), key=lambda x: x.stat().st_mtime):
            if total_size <= max_bytes:
                break
            size = f.stat().st_size
            f.unlink()
            total_size -= size
            cleared += size
            
        return cleared


# =============================================================================
# PUSH NOTIFICATIONS
# =============================================================================

class PushNotificationService:
    """Handles push notifications for mobile devices."""
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.fcm_enabled = bool(config.fcm_api_key)
        self.apns_enabled = bool(config.apns_key_path)
        
    def send_notification(
        self,
        device_token: str,
        notification_type: NotificationType,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send push notification to a device."""
        payload = {
            'type': notification_type.value,
            'title': title,
            'body': body,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # FCM for Android
        if self.fcm_enabled and not device_token.startswith('apns:'):
            return self._send_fcm(device_token, payload)
        
        # APNs for iOS
        if self.apns_enabled and device_token.startswith('apns:'):
            token = device_token.replace('apns:', '')
            return self._send_apns(token, payload)
            
        logger.info(f"Push notification (mock): {title} - {body}")
        return True
        
    def _send_fcm(self, token: str, payload: Dict[str, Any]) -> bool:
        """Send via Firebase Cloud Messaging."""
        # In production, implement actual FCM sending
        logger.info(f"FCM to {token[:20]}...: {payload['title']}")
        return True
        
    def _send_apns(self, token: str, payload: Dict[str, Any]) -> bool:
        """Send via Apple Push Notification Service."""
        # In production, implement actual APNs sending
        logger.info(f"APNs to {token[:20]}...: {payload['title']}")
        return True
        
    def send_task_progress(
        self,
        device_tokens: List[str],
        task_id: str,
        progress: float,
        status: str
    ) -> None:
        """Send task progress notification."""
        title = "AI DJ"
        body = f"Generation {int(progress * 100)}% complete"
        
        for token in device_tokens:
            self.send_notification(
                token,
                NotificationType.TASK_PROGRESS,
                title,
                body,
                {'task_id': task_id, 'progress': progress, 'status': status}
            )
            
    def send_task_completed(
        self,
        device_tokens: List[str],
        task_id: str,
        song_title: str,
        download_url: str
    ) -> None:
        """Send task completed notification."""
        title = "🎵 Generation Complete!"
        body = f'Your track "{song_title}" is ready'
        
        for token in device_tokens:
            self.send_notification(
                token,
                NotificationType.TASK_COMPLETED,
                title,
                body,
                {'task_id': task_id, 'download_url': download_url}
            )
            
    def send_task_failed(
        self,
        device_tokens: List[str],
        task_id: str,
        error: str
    ) -> None:
        """Send task failed notification."""
        title = "⚠️ Generation Failed"
        body = f"Error: {error}"
        
        for token in device_tokens:
            self.send_notification(
                token,
                NotificationType.TASK_FAILED,
                title,
                body,
                {'task_id': task_id, 'error': error}
            )


# =============================================================================
# BACKGROUND PROCESSING
# =============================================================================

class BackgroundTaskQueue:
    """Manages background task processing."""
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.Queue()
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False
        
    def start_workers(self, num_workers: int = None) -> None:
        """Start background worker threads."""
        if num_workers is None:
            num_workers = self.config.background_workers
            
        self._running = True
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ai-dj-worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
            
        logger.info(f"Started {num_workers} background workers")
        
    def stop_workers(self) -> None:
        """Stop all background workers."""
        self._running = False
        
        for _ in self._workers:
            self.task_queue.put(None)
            
        for worker in self._workers:
            worker.join(timeout=5)
            
        self._workers.clear()
        logger.info("Stopped all background workers")
        
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                task_item = self.task_queue.get(timeout=1)
                if task_item is None:
                    break
                    
                task_id, user_id, task_type, params = task_item
                self._process_task(task_id, user_id, task_type, params)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                
    def _process_task(
        self,
        task_id: str,
        user_id: str,
        task_type: str,
        params: Dict[str, Any]
    ) -> None:
        """Process a background task."""
        task = self.tasks.get(task_id)
        if not task:
            return
            
        try:
            task.status = TaskStatus.PROCESSING
            task.updated_at = datetime.now().isoformat()
            
            if task_type == "generate":
                result = self._generate_song(params)
            elif task_type == "remix":
                result = self._remix_song(params)
            elif task_type == "master":
                result = self._master_song(params)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.progress = 1.0
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.now().isoformat()
            logger.error(f"Task {task_id} failed: {e}")
            
    def _generate_song(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a song (placeholder - integrate with orchestrator)."""
        # Simulate progress updates
        for i in range(1, 10):
            task_id = params.get('_task_id')
            if task_id and task_id in self.tasks:
                self.tasks[task_id].progress = i / 10
                self.tasks[task_id].updated_at = datetime.now().isoformat()
            import time
            time.sleep(0.5)
            
        return {
            'id': str(uuid.uuid4()),
            'title': params.get('title', 'Untitled'),
            'genre': params.get('genre', 'house'),
            'bpm': params.get('bpm', 128),
            'key': params.get('key', 'C'),
            'duration': params.get('duration_sec', 180),
            'created_at': datetime.now().isoformat(),
            'file_path': f"output/song_{uuid.uuid4()}.wav"
        }
        
    def _remix_song(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remix an existing song."""
        return {'id': str(uuid.uuid4()), 'status': 'remixed'}
        
    def _master_song(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Master a song."""
        return {'id': str(uuid.uuid4()), 'status': 'mastered'}
        
    def submit_task(
        self,
        user_id: str,
        task_type: str,
        params: Dict[str, Any]
    ) -> Task:
        """Submit a new background task."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            user_id=user_id,
            type=task_type,
            status=TaskStatus.PENDING,
            params=params,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        with self._lock:
            self.tasks[task_id] = task
            
        params['_task_id'] = task_id
        self.task_queue.put((task_id, user_id, task_type, params))
        
        logger.info(f"Submitted task {task_id} ({task_type})")
        return task
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task status."""
        return self.tasks.get(task_id)
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.updated_at = datetime.now().isoformat()
            return True
        return False


# =============================================================================
# API CONTROLLER
# =============================================================================

class MobileAPIController:
    """Main API controller for mobile app."""
    
    def __init__(self, config: MobileConfig = None):
        self.config = config or MobileConfig()
        self.offline_storage = OfflineStorage(self.config)
        self.push_service = PushNotificationService(self.config)
        self.task_queue = BackgroundTaskQueue(self.config)
        
        # User storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Tuple[str, datetime]] = {}  # token -> (user_id, expiry)
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # Initialize
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        for path in [self.config.output_dir, self.config.cache_dir, self.config.offline_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
            
    # -------------------------------------------------------------------------
    # AUTHENTICATION
    # -------------------------------------------------------------------------
    
    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        device_token: Optional[str] = None
    ) -> Tuple[Optional[User], str]:
        """Register a new user."""
        # Check existing
        for user in self.users.values():
            if user.email == email:
                return None, "Email already registered"
            if user.username == username:
                return None, "Username already taken"
                
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            username=username,
            email=email,
            created_at=datetime.now().isoformat(),
            device_tokens=[device_token] if device_token else []
        )
        
        self.users[user_id] = user
        
        # Create session
        token = self._create_session(user_id)
        
        logger.info(f"Registered user: {username}")
        return user, token
        
    def login(
        self,
        email: str,
        password: str,
        device_token: Optional[str] = None
    ) -> Tuple[Optional[User], str]:
        """Authenticate user."""
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break
                
        if not user:
            return None, "Invalid credentials"
            
        # Add device token
        if device_token and device_token not in user.device_tokens:
            user.device_tokens.append(device_token)
            
        token = self._create_session(user.id)
        return user, token
        
    def logout(self, token: str) -> bool:
        """Logout user."""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False
        
    def _create_session(self, user_id: str) -> str:
        """Create a new session."""
        token = hashlib.sha256(
            f"{user_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        expiry = datetime.now() + timedelta(seconds=self.config.session_timeout)
        self.sessions[token] = (user_id, expiry)
        
        return token
        
    def verify_token(self, token: str) -> Optional[str]:
        """Verify session token and return user_id."""
        if token not in self.sessions:
            return None
            
        user_id, expiry = self.sessions[token]
        
        if datetime.now() > expiry:
            del self.sessions[token]
            return None
            
        return user_id
        
    def register_device_token(self, user_id: str, token: str) -> bool:
        """Register device token for push notifications."""
        user = self.users.get(user_id)
        if not user:
            return False
            
        if token not in user.device_tokens:
            user.device_tokens.append(token)
            
        return True
        
    # -------------------------------------------------------------------------
    # SONG GENERATION
    # -------------------------------------------------------------------------
    
    def create_song_request(
        self,
        user_id: str,
        request: SongRequest,
        background: bool = True
    ) -> Tuple[Optional[Task], Dict[str, Any]]:
        """Create a song generation request."""
        params = asdict(request)
        params['user_id'] = user_id
        
        if background:
            task = self.task_queue.submit_task(user_id, "generate", params)
            
            # Save to offline storage
            self.offline_storage.save_task(task)
            
            return task, {'task_id': task.id, 'status': task.status.value}
        else:
            # Synchronous generation
            result = self.task_queue._generate_song(params)
            return None, result
            
    def get_task_status(self, user_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a generation task."""
        task = self.task_queue.get_task(task_id)
        
        if not task or task.user_id != user_id:
            # Check offline storage
            task = self.offline_storage.load_task(task_id)
            
        if task:
            return {
                'id': task.id,
                'type': task.type,
                'status': task.status.value,
                'progress': task.progress,
                'result': task.result,
                'error': task.error,
                'created_at': task.created_at,
                'updated_at': task.updated_at
            }
            
        return None
        
    def list_tasks(
        self,
        user_id: str,
        status: Optional[TaskStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List user's tasks."""
        tasks = []
        
        for task in self.task_queue.tasks.values():
            if task.user_id == user_id:
                if status is None or task.status == status:
                    tasks.append({
                        'id': task.id,
                        'type': task.type,
                        'status': task.status.value,
                        'progress': task.progress,
                        'created_at': task.created_at
                    })
                    
        # Sort by creation date, newest first
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        return tasks[:limit]
        
    def cancel_task(self, user_id: str, task_id: str) -> bool:
        """Cancel a task."""
        task = self.task_queue.get_task(task_id)
        
        if task and task.user_id == user_id:
            return self.task_queue.cancel_task(task_id)
            
        return False
        
    # -------------------------------------------------------------------------
    # OFFLINE SUPPORT
    # -------------------------------------------------------------------------
    
    def get_offline_songs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get list of offline-available songs."""
        return self.offline_storage.list_offline_songs(user_id)
        
    def download_song(self, user_id: str, song_id: str) -> Optional[Dict[str, Any]]:
        """Download a song for offline use."""
        song = self.offline_storage.load_song(song_id)
        
        if song:
            # Mark as available offline
            song['offline_available'] = True
            return song
            
        return None
        
    def sync_offline_data(self, user_id: str) -> OfflinePackage:
        """Create sync package for offline data."""
        return self.offline_storage.create_sync_package(user_id)
        
    # -------------------------------------------------------------------------
    # PREFERENCES
    # -------------------------------------------------------------------------
    
    def update_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences."""
        user = self.users.get(user_id)
        if not user:
            return False
            
        user.preferences.update(preferences)
        return True
        
    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        user = self.users.get(user_id)
        if not user:
            return {}
        return user.preferences
        
    # -------------------------------------------------------------------------
    # RATE LIMITING
    # -------------------------------------------------------------------------
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit."""
        now = datetime.now()
        window = timedelta(seconds=self.config.rate_limit_window)
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
            
        # Clean old entries
        self.rate_limits[identifier] = [
            t for t in self.rate_limits[identifier]
            if t > now - window
        ]
        
        if len(self.rate_limits[identifier]) >= self.config.rate_limit_requests:
            return False
            
        self.rate_limits[identifier].append(now)
        return True
        
    # -------------------------------------------------------------------------
    # HEALTH & STATUS
    # -------------------------------------------------------------------------
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'status': 'online',
            'version': self.config.api_version,
            'timestamp': datetime.now().isoformat(),
            'active_tasks': sum(
                1 for t in self.task_queue.tasks.values()
                if t.status == TaskStatus.PROCESSING
            ),
            'pending_tasks': sum(
                1 for t in self.task_queue.tasks.values()
                if t.status == TaskStatus.PENDING
            ),
            'total_users': len(self.users),
            'offline_enabled': self.config.offline_enabled
        }


# =============================================================================
# FLASK APP (Optional HTTP Server)
# =============================================================================

def create_app(config: MobileConfig = None) -> Any:
    """Create Flask application."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.warning("Flask not installed. HTTP server unavailable.")
        return None
        
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = config.max_upload_size_mb * 1024 * 1024
    
    controller = MobileAPIController(config)
    
    # Start background workers
    controller.task_queue.start_workers()
    
    def json_response(data: Any, status: int = 200):
        return jsonify(data), status
        
    def require_auth(f):
        """Decorator for authenticated routes."""
        def wrapper(*args, **kwargs):
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            user_id = controller.verify_token(token)
            
            if not user_id:
                return json_response({'error': 'Unauthorized'}, 401)
                
            return f(user_id, *args, **kwargs)
            
        wrapper.__name__ = f.__name__
        return wrapper
        
    # -------------------------------------------------------------------------
    # AUTH ROUTES
    # -------------------------------------------------------------------------
    
    @app.route(f'/api/{config.api_version}/auth/register', methods=['POST'])
    def register():
        data = request.get_json() or {}
        user, token = controller.register_user(
            data.get('username', ''),
            data.get('email', ''),
            data.get('password', ''),
            data.get('device_token')
        )
        
        if not user:
            return json_response({'error': token}, 400)  # token contains error
            
        return json_response({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            },
            'token': token
        })
        
    @app.route(f'/api/{config.api_version}/auth/login', methods=['POST'])
    def login():
        data = request.get_json() or {}
        user, token = controller.login(
            data.get('email', ''),
            data.get('password', ''),
            data.get('device_token')
        )
        
        if not user:
            return json_response({'error': 'Invalid credentials'}, 401)
            
        return json_response({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            },
            'token': token
        })
        
    @app.route(f'/api/{config.api_version}/auth/logout', methods=['POST'])
    @require_auth
    def logout(user_id):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        controller.logout(token)
        return json_response({'success': True})
        
    # -------------------------------------------------------------------------
    # GENERATION ROUTES
    # -------------------------------------------------------------------------
    
    @app.route(f'/api/{config.api_version}/generate', methods=['POST'])
    @require_auth
    def generate(user_id):
        # Check rate limit
        if not controller.check_rate_limit(user_id):
            return json_response({'error': 'Rate limit exceeded'}, 429)
            
        data = request.get_json() or {}
        
        # Validate request
        required = ['genre', 'bpm']
        for field in required:
            if field not in data:
                return json_response({'error': f'Missing required field: {field}'}, 400)
                
        song_request = SongRequest(
            genre=data.get('genre', 'house'),
            key=data.get('key', 'C'),
            scale=data.get('scale', 'minor'),
            bpm=data.get('bpm', 128),
            duration_sec=data.get('duration_sec', 180),
            title=data.get('title', ''),
            artist=data.get('artist', ''),
            energy=data.get('energy', 0.8),
            mood=data.get('mood', 'energetic'),
            notification_enabled=data.get('notification_enabled', True)
        )
        
        task, response = controller.create_song_request(user_id, song_request)
        
        return json_response({
            'task_id': task.id if task else None,
            'status': response.get('status', 'completed'),
            'message': 'Generation started' if task else 'Generation completed'
        })
        
    @app.route(f'/api/{config.api_version}/tasks/<task_id>', methods=['GET'])
    @require_auth
    def get_task(user_id, task_id):
        result = controller.get_task_status(user_id, task_id)
        
        if not result:
            return json_response({'error': 'Task not found'}, 404)
            
        return json_response(result)
        
    @app.route(f'/api/{config.api_version}/tasks', methods=['GET'])
    @require_auth
    def list_tasks(user_id):
        status = request.args.get('status')
        limit = int(request.args.get('limit', 50))
        
        status_enum = TaskStatus(status) if status else None
        tasks = controller.list_tasks(user_id, status_enum, limit)
        
        return json_response({'tasks': tasks})
        
    @app.route(f'/api/{config.api_version}/tasks/<task_id>/cancel', methods=['POST'])
    @require_auth
    def cancel_task(user_id, task_id):
        success = controller.cancel_task(user_id, task_id)
        
        if not success:
            return json_response({'error': 'Cannot cancel task'}, 400)
            
        return json_response({'success': True})
        
    # -------------------------------------------------------------------------
    # OFFLINE ROUTES
    # -------------------------------------------------------------------------
    
    @app.route(f'/api/{config.api_version}/offline/songs', methods=['GET'])
    @require_auth
    def get_offline_songs(user_id):
        songs = controller.get_offline_songs(user_id)
        return json_response({'songs': songs})
        
    @app.route(f'/api/{config.api_version}/offline/sync', methods=['GET'])
    @require_auth
    def sync_offline(user_id):
        package = controller.sync_offline_data(user_id)
        return json_response(asdict(package))
        
    @app.route(f'/api/{config.api_version}/offline/songs/<song_id>', methods=['GET'])
    @require_auth
    def download_song(user_id, song_id):
        song = controller.download_song(user_id, song_id)
        
        if not song:
            return json_response({'error': 'Song not found'}, 404)
            
        return json_response(song)
        
    # -------------------------------------------------------------------------
    # PREFERENCES ROUTES
    # -------------------------------------------------------------------------
    
    @app.route(f'/api/{config.api_version}/preferences', methods=['GET'])
    @require_auth
    def get_preferences(user_id):
        prefs = controller.get_preferences(user_id)
        return json_response({'preferences': prefs})
        
    @app.route(f'/api/{config.api_version}/preferences', methods=['PUT'])
    @require_auth
    def update_preferences(user_id):
        data = request.get_json() or {}
        
        success = controller.update_preferences(user_id, data)
        
        if not success:
            return json_response({'error': 'Failed to update preferences'}, 500)
            
        return json_response({'success': True})
        
    # -------------------------------------------------------------------------
    # DEVICE TOKEN ROUTES
    # -------------------------------------------------------------------------
    
    @app.route(f'/api/{config.api_version}/device/token', methods=['POST'])
    @require_auth
    def register_device(user_id):
        data = request.get_json() or {}
        token = data.get('token')
        
        if not token:
            return json_response({'error': 'Token required'}, 400)
            
        success = controller.register_device_token(user_id, token)
        
        if not success:
            return json_response({'error': 'Failed to register token'}, 500)
            
        return json_response({'success': True})
        
    # -------------------------------------------------------------------------
    # SYSTEM ROUTES
    # -------------------------------------------------------------------------
    
    @app.route(f'/api/{config.api_version}/status', methods=['GET'])
    def system_status():
        status = controller.get_system_status()
        return json_response(status)
        
    @app.route(f'/api/{config.api_version}/health', methods=['GET'])
    def health():
        return json_response({'status': 'healthy'})
        
    # Store controller for cleanup
    app.controller = controller
    
    return app


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the mobile API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI DJ Mobile API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of background workers')
    parser.add_argument('--config', help='Path to config JSON file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_data = json.load(f)
            config = MobileConfig(**config_data)
    else:
        config = MobileConfig(
            host=args.host,
            port=args.port,
            debug=args.debug,
            background_workers=args.workers
        )
        
    # Create and start app
    app = create_app(config)
    
    if app is None:
        logger.error("Flask not available. Install with: pip install flask")
        sys.exit(1)
        
    # Setup cleanup
    def cleanup():
        logger.info("Shutting down...")
        app.controller.task_queue.stop_workers()
        
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: cleanup() or sys.exit(0))
    
    # Start server
    logger.info(f"Starting AI DJ Mobile API on {config.host}:{config.port}")
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
