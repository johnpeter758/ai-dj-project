#!/usr/bin/env python3
"""
Security Module - Authentication & Encryption

Provides authentication, encryption, and secure data handling
for the AI DJ Project.
"""

import os
import hashlib
import hmac
import secrets
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# Configuration
SECRET_KEY_ENV = "AI_DJ_SECRET_KEY"
API_KEY_ENV = "AI_DJ_API_KEY"
SESSION_TIMEOUT_MINUTES = 60


@dataclass
class AuthToken:
    """Authentication token container"""
    token: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    permissions: list = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        return datetime.now() < self.expires_at
    
    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "permissions": self.permissions,
        }


class SecurityError(Exception):
    """Base exception for security errors"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass


class EncryptionError(SecurityError):
    """Encryption/decryption failed"""
    pass


class AuthManager:
    """Manages authentication and API keys"""
    
    def __init__(self):
        self._api_keys: Dict[str, Dict] = {}
        self._tokens: Dict[str, AuthToken] = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment"""
        api_key = os.environ.get(API_KEY_ENV)
        if api_key:
            self._api_keys[api_key] = {
                "created": datetime.now(),
                "permissions": ["full"],
            }
    
    def generate_api_key(self, user_id: str, permissions: list = None) -> str:
        """Generate a new API key"""
        key = f"aidj_{secrets.token_urlsafe(32)}"
        self._api_keys[key] = {
            "user_id": user_id,
            "created": datetime.now(),
            "permissions": permissions or ["read"],
        }
        return key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key"""
        if not api_key or api_key not in self._api_keys:
            return False
        return True
    
    def get_key_permissions(self, api_key: str) -> Optional[list]:
        """Get permissions for an API key"""
        if api_key in self._api_keys:
            return self._api_keys[api_key].get("permissions", [])
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            return True
        return False
    
    def create_session(self, user_id: str, permissions: list = None) -> AuthToken:
        """Create a new session token"""
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        
        auth_token = AuthToken(
            token=token,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=SESSION_TIMEOUT_MINUTES),
            permissions=permissions or ["read"],
        )
        
        self._tokens[token] = auth_token
        return auth_token
    
    def validate_session(self, token: str) -> Optional[AuthToken]:
        """Validate a session token"""
        if token in self._tokens:
            auth_token = self._tokens[token]
            if auth_token.is_valid():
                return auth_token
            else:
                # Clean up expired token
                del self._tokens[token]
        return None
    
    def refresh_session(self, token: str) -> Optional[AuthToken]:
        """Refresh a session token"""
        old_token = self.validate_session(token)
        if old_token:
            # Create new token with same permissions
            return self.create_session(
                old_token.user_id,
                old_token.permissions
            )
        return None
    
    def revoke_session(self, token: str) -> bool:
        """Revoke a session token"""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False
    
    @staticmethod
    def hash_password(password: str, salt: bytes = None) -> tuple:
        """Hash a password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return base64.b64encode(key).decode('utf-8'), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: bytes) -> bool:
        """Verify a password against its hash"""
        computed_hash, _ = SecurityManager.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, hashed)


class EncryptionManager:
    """Manages encryption/decryption of sensitive data"""
    
    def __init__(self, key: bytes = None):
        if not CRYPTO_AVAILABLE:
            raise EncryptionError(
                "cryptography library not installed. "
                "Run: pip install cryptography"
            )
        
        if key is None:
            key = self._get_or_create_key()
        
        self._fernet = Fernet(key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        key_path = Path.home() / ".ai_dj" / ".key"
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        
        # Generate new key
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key)
        os.chmod(key_path, 0o600)
        
        return key
    
    @classmethod
    def derive_key_from_password(cls, password: str, salt: bytes = None) -> bytes:
        """Derive an encryption key from a password"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        try:
            encrypted = self._fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt: {e}")
    
    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        """Encrypt a file"""
        input_p = Path(input_path)
        
        if not input_p.exists():
            raise EncryptionError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = str(input_p) + ".enc"
        
        with open(input_p, 'rb') as f:
            data = f.read()
        
        encrypted = self._fernet.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        """Decrypt a file"""
        input_p = Path(input_path)
        
        if not input_p.exists():
            raise EncryptionError(f"Encrypted file not found: {input_path}")
        
        with open(input_p, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self._fernet.decrypt(encrypted)
        
        if output_path is None:
            output_path = str(input_p).replace(".enc", ".dec")
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        
        return output_path


class SecurityManager:
    """High-level security operations"""
    
    def __init__(self):
        self.auth = AuthManager()
        self._encryption: Optional[EncryptionManager] = None
    
    @property
    def encryption(self) -> EncryptionManager:
        """Lazy-load encryption manager"""
        if self._encryption is None:
            try:
                self._encryption = EncryptionManager()
            except EncryptionError:
                # Cryptography not available - will return None
                pass
        return self._encryption
    
    def secure_api_call(self, api_key: str, required_permission: str = None):
        """Decorator for securing API endpoints"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Validate API key
                if not self.auth.validate_api_key(api_key):
                    raise AuthenticationError("Invalid API key")
                
                # Check permissions if required
                if required_permission:
                    perms = self.auth.get_key_permissions(api_key)
                    if required_permission not in perms:
                        raise AuthenticationError(
                            f"Missing required permission: {required_permission}"
                        )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_data(data: str) -> str:
        """Generate SHA-256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def verify_signature(data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature"""
        expected = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    
    @staticmethod
    def generate_signature(data: str, secret: str) -> str:
        """Generate HMAC signature"""
        return hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()


# Convenience instances
security = SecurityManager()
auth = AuthManager()
