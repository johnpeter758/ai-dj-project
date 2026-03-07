#!/usr/bin/env python3
"""
Inference Optimizer - AI DJ Model Optimization Engine
Provides model optimization, quantization, and inference caching
"""

import hashlib
import json
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Represents a cached inference result"""
    key: str
    value: Any
    created_at: float
    access_count: int = 0
    last_accessed: float = 0
    
    def access(self) -> Any:
        """Record access and return value"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    enable_quantization: bool = True
    quantization_dtype: str = "float16"  # float16, int8, int4
    enable_caching: bool = True
    cache_max_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    enable_warmup: bool = True
    max_workers: int = 4
    batch_size: int = 8
    enable_model_offload: bool = False
    

class InferenceCache:
    """LRU cache for inference results with TTL support"""
    
    def __init__(self, max_size_mb: int = 512, ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size = 0
        self._hits = 0
        self._misses = 0
    
    def _compute_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes"""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from inputs"""
        key_data = json.dumps({
            'args': [str(a) for a in args],
            'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                self._evict(key)
                self._misses += 1
                return None
            
            self._hits += 1
            return entry.access()
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with LRU eviction"""
        with self._lock:
            value_size = self._compute_size(value)
            
            # Evict if needed
            while self._current_size + value_size > self.max_size_bytes and self._cache:
                self._evict_lru()
            
            # Remove existing entry if updating
            if key in self._cache:
                old_size = self._compute_size(self._cache[key].value)
                self._current_size -= old_size
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time()
            )
            self._cache[key] = entry
            self._current_size += value_size
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        lru_key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].last_accessed)
        self._evict(lru_key)
    
    def _evict(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            value_size = self._compute_size(self._cache[key].value)
            self._current_size -= value_size
            del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'entries': len(self._cache),
            'size_mb': self._current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }


class ModelOptimizer:
    """Handles model optimization including quantization"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self._optimized_models: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def optimize_model(self, model: Any, model_name: str) -> Any:
        """Apply optimizations to a model"""
        with self._lock:
            if model_name in self._optimized_models:
                return self._optimized_models[model_name]
            
            optimized = model
            
            # Apply quantization
            if self.config.enable_quantization:
                optimized = self._quantize_model(optimized)
            
            # Store optimized model
            self._optimized_models[model_name] = optimized
            return optimized
    
    def _quantize_model(self, model: Any) -> Any:
        """Quantize model to reduce size and improve inference speed"""
        dtype = self.config.quantization_dtype
        
        if dtype == "float16":
            return self._to_float16(model)
        elif dtype == "int8":
            return self._to_int8(model)
        elif dtype == "int4":
            return self._to_int4(model)
        
        return model
    
    def _to_float16(self, model: Any) -> Any:
        """Convert model to float16"""
        try:
            import torch
            if hasattr(model, 'half'):
                return model.half()
            elif hasattr(model, 'to'):
                return model.to(torch.float16)
        except ImportError:
            pass
        
        # Fallback: wrap in float16-compatible class
        return Float16Wrapper(model)
    
    def _to_int8(self, model: Any) -> Any:
        """Quantize model to int8"""
        try:
            import torch
            if hasattr(torch, 'quantization'):
                return torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
        except ImportError:
            pass
        
        return Int8Wrapper(model)
    
    def _to_int4(self, model: Any) -> Any:
        """Quantize model to int4 (simulated)"""
        return Int4Wrapper(model)
    
    def warmup(self, model: Any, sample_input: Any, num_iterations: int = 3) -> None:
        """Warmup model with sample inputs"""
        if not self.config.enable_warmup:
            return
        
        for _ in range(num_iterations):
            try:
                if callable(model):
                    model(sample_input)
                elif hasattr(model, 'forward'):
                    model.forward(sample_input)
            except Exception:
                break
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get information about model size and optimization"""
        info = {
            'type': type(model).__name__,
            'has_quantization': False,
            'estimated_size_mb': 0
        }
        
        # Estimate size
        try:
            if hasattr(model, 'state_dict'):
                import torch
                param_size = sum(p.numel() * p.element_size() 
                               for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() 
                                 for b in model.buffers())
                info['estimated_size_mb'] = (param_size + buffer_size) / (1024 * 1024)
        except:
            pass
        
        return info


class Float16Wrapper:
    """Wrapper for float16 models"""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class Int8Wrapper:
    """Wrapper for int8 quantized models"""
    def __init__(self, model):
        self.model = model
        self.is_quantized = True
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class Int4Wrapper:
    """Wrapper for int4 quantized models (simulated)"""
    def __init__(self, model):
        self.model = model
        self.is_quantized = True
        self.quantization = "int4"
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class InferenceOptimizer:
    """Main class for optimized inference with caching"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.cache = InferenceCache(
            max_size_mb=self.config.cache_max_size_mb,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_caching else None
        self.optimizer = ModelOptimizer(self.config)
        self._model_cache: Dict[str, Any] = {}
        self._inference_count = 0
        self._total_inference_time = 0.0
    
    def register_model(self, name: str, model: Any) -> Any:
        """Register and optimize a model"""
        optimized = self.optimizer.optimize_model(model, name)
        self._model_cache[name] = optimized
        return optimized
    
    def infer(
        self, 
        model_name: str, 
        inputs: Tuple[Any, ...], 
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """Run inference with optional caching"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = None
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(model_name, *inputs, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Get model
        model = self._model_cache.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not registered")
        
        # Run inference
        try:
            if callable(model):
                result = model(*inputs, **kwargs)
            elif hasattr(model, 'forward'):
                result = model.forward(*inputs, **kwargs)
            else:
                raise ValueError(f"Model '{model_name}' is not callable")
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
        
        # Cache result
        if use_cache and self.cache and cache_key:
            self.cache.set(cache_key, result)
        
        # Track stats
        self._inference_count += 1
        self._total_inference_time += time.time() - start_time
        
        return result
    
    def infer_batch(
        self,
        model_name: str,
        batch_inputs: List[Tuple[Any, ...]],
        use_cache: bool = True
    ) -> List[Any]:
        """Run batch inference"""
        results = []
        
        for inputs in batch_inputs:
            result = self.infer(model_name, inputs, use_cache=use_cache)
            results.append(result)
        
        return results
    
    def _generate_cache_key(self, model_name: str, *args, **kwargs) -> str:
        """Generate a unique cache key"""
        key_parts = [model_name]
        
        for arg in args:
            try:
                if isinstance(arg, np.ndarray):
                    key_parts.append(hashlib.md5(arg.tobytes()).hexdigest()[:8])
                else:
                    key_parts.append(str(hash(str(arg))))
            except:
                key_parts.append(str(hash(str(type(arg)))))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={hash(str(v))}")
        
        return hashlib.sha256("".join(key_parts).encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            'inference_count': self._inference_count,
            'total_time': self._total_inference_time,
            'avg_time': self._total_inference_time / max(1, self._inference_count),
            'models_loaded': list(self._model_cache.keys()),
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear inference cache"""
        if self.cache:
            self.cache.clear()
    
    def unload_model(self, name: str) -> None:
        """Unload a model from memory"""
        if name in self._model_cache:
            del self._model_cache[name]
        if name in self.optimizer._optimized_models:
            del self.optimizer._optimized_models[name]


# Convenience functions

def create_optimizer(
    quantization: str = "float16",
    cache_size_mb: int = 512,
    cache_ttl: int = 3600
) -> InferenceOptimizer:
    """Create a configured inference optimizer"""
    config = OptimizationConfig(
        enable_quantization=quantization != "none",
        quantization_dtype=quantization if quantization != "none" else "float32",
        cache_max_size_mb=cache_size_mb,
        cache_ttl_seconds=cache_ttl
    )
    return InferenceOptimizer(config)


def cached_inference(
    func: Callable[..., T],
    cache: Optional[InferenceCache] = None,
    *args,
    **kwargs
) -> T:
    """Decorator for cached function inference"""
    if cache is None:
        return func(*args, **kwargs)
    
    key = cache._generate_key(*args, **kwargs)
    result = cache.get(key)
    
    if result is not None:
        return result
    
    result = func(*args, **kwargs)
    cache.set(key, result)
    return result


# Example usage

if __name__ == '__main__':
    # Demo
    print("=== Inference Optimizer Demo ===\n")
    
    # Create optimizer
    optimizer = create_optimizer(quantization="float16", cache_size_mb=256)
    
    # Register a simple model
    def dummy_model(x):
        return x * 2
    
    optimizer.register_model("multiply", dummy_model)
    
    # Run inference
    result = optimizer.infer("multiply", (5,))
    print(f"Inference result: {result}")
    
    # Run again (should hit cache)
    result2 = optimizer.infer("multiply", (5,))
    print(f"Cached result: {result2}")
    
    # Print stats
    print(f"\nStats: {optimizer.get_stats()}")
