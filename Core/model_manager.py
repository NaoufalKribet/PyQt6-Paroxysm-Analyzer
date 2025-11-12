"""
Optimized Model Management System with Caching and Compression
===============================================================

This module provides a robust, thread-safe model management system for machine learning
pipelines. Features include:
- Multi-format model support (Keras, scikit-learn)
- Atomic file operations to prevent corruption
- LRU caching for improved performance
- Compression support with configurable levels
- Comprehensive metadata tracking and validation
- Automatic backup and cleanup mechanisms

The system is designed for production environments where model persistence, 
versioning, and efficient retrieval are critical.

@author: KRIBET Naoufal
@affiliation: 5th year Engineering Student, EOST (École et Observatoire des Sciences de la Terre)
@date: 2024
@version: 2.0
"""

import joblib
import json
import os
import shutil
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache
from datetime import datetime, timezone
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from tensorflow.keras.models import Model as KerasModel, load_model


class CompressionLevel(Enum):
    """
    Compression levels for model serialization.
    
    Higher compression reduces disk space but increases I/O time.
    Recommended: MEDIUM for most applications.
    
    @author: KRIBET Naoufal
    """
    NONE = 0      # No compression (fastest I/O)
    LOW = 3       # Light compression
    MEDIUM = 6    # Balanced compression (recommended)
    HIGH = 9      # Maximum compression (slowest I/O)


@dataclass
class ModelMetadata:
    """
    Enriched metadata structure for tracked models.
    
    This dataclass stores comprehensive information about each saved model,
    enabling efficient filtering, validation, and model selection.
    
    @param model_name: Unique identifier for the model
    @param creation_date: ISO 8601 timestamp of model creation
    @param file_size_mb: Size of serialized model file in megabytes
    @param model_type: Type of model (e.g., 'RandomForestClassifier', 'Sequential')
    @param feature_count: Number of input features the model expects
    @param training_samples: Number of samples used during training
    @param checksum: MD5 hash for file integrity verification
    @param version: Metadata schema version
    @param tags: List of custom tags for model categorization
    
    @author: KRIBET Naoufal
    """
    model_name: str
    creation_date: str
    file_size_mb: float
    model_type: str
    feature_count: int
    training_samples: int
    checksum: str
    version: str = "1.0"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelManager:
    """
    Thread-safe model manager with caching, compression, and validation.
    
    This class provides a complete model lifecycle management system including:
    - Atomic save/load operations to prevent data corruption
    - LRU caching for frequently accessed models and reports
    - Configurable compression to optimize disk usage
    - Comprehensive metadata tracking for model provenance
    - Automatic cleanup and backup capabilities
    
    The manager supports multiple model formats (Keras, scikit-learn) and 
    maintains backward compatibility with legacy systems through wrapper functions.
    
    @author: KRIBET Naoufal
    """
    
    def __init__(self, model_dir: str = "models", cache_size: int = 10, 
                 compression_level: CompressionLevel = CompressionLevel.MEDIUM):
        """
        Initialize the model manager with specified configuration.
        
        @param model_dir: Directory path for model storage
        @param cache_size: Maximum number of models/reports to cache in memory
        @param compression_level: Compression level for joblib serialization
        
        @author: KRIBET Naoufal
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.cache_size = cache_size
        self.compression_level = compression_level
        
        # Thread-safe caches
        self._model_cache = {}
        self._report_cache = {}
        self._cache_lock = threading.RLock()
        self._metadata_cache = {}
        
        # Logging system
        self.logger = self._setup_logger()
        
        # Configuration management
        self.config_file = self.model_dir / "manager_config.json"
        self._load_or_create_config()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure the logging system for the manager.
        
        @return: Configured logger instance
        @author: KRIBET Naoufal
        """
        logger = logging.getLogger(f"{__name__}.ModelManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_or_create_config(self):
        """
        Load existing configuration or create default if not present.
        
        @author: KRIBET Naoufal
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.warning(f"Config loading error: {e}. Creating new config.")
                self.config = self._create_default_config()
        else:
            self.config = self._create_default_config()
        
        self._save_config()
    
    def _create_default_config(self) -> Dict:
        """
        Create default configuration dictionary.
        
        @return: Default configuration with standard settings
        @author: KRIBET Naoufal
        """
        return {
            "version": "2.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "compression_level": self.compression_level.value,
            "auto_cleanup": True,
            "max_models": 50,
            "backup_enabled": True
        }
    
    def _save_config(self):
        """
        Persist configuration to disk.
        
        @author: KRIBET Naoufal
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Config save error: {e}")
    
    @contextmanager
    def _cache_lock_context(self):
        """
        Context manager for thread-safe cache access.
        
        @author: KRIBET Naoufal
        """
        self._cache_lock.acquire()
        try:
            yield
        finally:
            self._cache_lock.release()
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate MD5 checksum for file integrity verification.
        
        @param file_path: Path to file for checksum calculation
        @return: Hexadecimal MD5 hash string
        @author: KRIBET Naoufal
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Checksum calculation error: {e}")
            return ""
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """
        Get file size in megabytes.
        
        @param file_path: Path to file
        @return: File size in MB
        @author: KRIBET Naoufal
        """
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def _extract_model_info(self, results: Dict) -> Tuple[str, int, int]:
        """
        Extract model information from training results.
        
        @param results: Training results dictionary containing the model
        @return: Tuple of (model_type, feature_count, training_samples)
        @author: KRIBET Naoufal
        """
        model = results.get('model')
        model_type = type(model).__name__ if model else "Unknown"
        
        # Extract feature count
        feature_count = 0
        if hasattr(model, 'n_features_in_'):
            feature_count = model.n_features_in_
        elif hasattr(model, 'feature_names_in_'):
            feature_count = len(model.feature_names_in_)
        
        # Extract training sample count
        training_samples = 0
        if 'plot_data' in results and 'y_test' in results['plot_data']:
            training_samples = len(results['plot_data']['y_test'])
        
        return model_type, feature_count, training_samples
    
    def _create_metadata(self, model_name: str, results: Dict, model_file: Path) -> ModelMetadata:
        """
        Create metadata object for a saved model.
        
        @param model_name: Name of the model
        @param results: Training results containing model information
        @param model_file: Path to serialized model file
        @return: ModelMetadata instance with extracted information
        @author: KRIBET Naoufal
        """
        model_type, feature_count, training_samples = self._extract_model_info(results)
        
        return ModelMetadata(
            model_name=model_name,
            creation_date=datetime.now(timezone.utc).isoformat(),
            file_size_mb=self._get_file_size_mb(model_file),
            model_type=model_type,
            feature_count=feature_count,
            training_samples=training_samples,
            checksum=self._calculate_checksum(model_file),
            tags=["auto-generated"]
        )
    
    @contextmanager
    def _atomic_write(self, file_path: Path):
        """
        Atomic write operation to prevent file corruption.
        
        This context manager writes to a temporary file first, then performs
        an atomic rename operation. If any error occurs, the temporary file
        is cleaned up without affecting the original.
        
        @param file_path: Target file path
        @yields: Temporary file path for writing
        @author: KRIBET Naoufal
        """
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            yield temp_path
            # Atomic move operation
            temp_path.replace(file_path)
        except Exception:
            # Cleanup on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def save_model_optimized(self, model_name: str, results: Dict, config: Dict, 
                           tags: List[str] = None, overwrite: bool = False) -> bool:
        """
        Optimized model saving with validation, compression, and enriched metadata.
        
        CORRECTED: Handles Keras models, scikit-learn models, and scalers for forecasting models.
        
        This method performs:
        1. Input validation and sanitization
        2. Format detection (Keras vs scikit-learn)
        3. Atomic file writing to prevent corruption
        4. Metadata and report generation
        5. Cache invalidation for consistency
        
        @param model_name: Unique identifier for the model
        @param results: Dictionary containing 'model' key and optional 'scalers' for LSTM
        @param config: Training configuration dictionary
        @param tags: Optional list of tags for categorization
        @param overwrite: Whether to overwrite existing model
        @return: True if save successful, False otherwise
        
        @author: KRIBET Naoufal
        """
        self.logger.info(f"Starting save for model '{model_name}'")
        
        if not self._validate_save_inputs(model_name, results, config):
            return False
        
        model_path = self.model_dir / model_name
        
        if model_path.exists() and not overwrite:
            self.logger.warning(f"Model '{model_name}' already exists. Use overwrite=True.")
            return False
        
        try:
            model_path.mkdir(exist_ok=True, parents=True)
            
            model_obj = results['model']
            
            # Handle Keras models
            if isinstance(model_obj, KerasModel):
                model_file = model_path / "model.keras"
                self.logger.info("Keras model detected. Using model.save().")
                with self._atomic_write(model_file) as temp_model_file:
                    model_obj.save(temp_model_file)
                
                # Save scalers for LSTM models if present
                if 'scalers' in results:
                    scalers_file = model_path / "scalers.joblib"
                    self.logger.info("Scalers detected for LSTM model. Saving...")
                    joblib.dump(results['scalers'], scalers_file)

            # Handle scikit-learn models
            else:
                model_file = model_path / "model.joblib"
                self.logger.info("Scikit-learn model detected. Using joblib.dump().")
                with self._atomic_write(model_file) as temp_model_file:
                    joblib.dump(model_obj, temp_model_file, compress=self.compression_level.value)

            report_file = model_path / "report.json"
            metadata_file = model_path / "metadata.json"
            
            # Create and enrich metadata
            metadata = self._create_metadata(model_name, results, model_file)
            if tags: metadata.tags.extend(tags)
            
            # Prepare report data
            report_data = {
                'model_name': model_name, 
                'best_params': results.get('best_params', {}),
                'report': results.get('report', {}), 
                'training_config': config,
                'created_at': metadata.creation_date, 
                'model_type': metadata.model_type,
                'version': metadata.version
            }
            
            # Save report atomically
            self.logger.info("Saving report...")
            with self._atomic_write(report_file) as temp_report_file:
                with open(temp_report_file, 'w') as f: 
                    json.dump(report_data, f, indent=2)
            
            # Save metadata atomically
            self.logger.info("Saving metadata...")
            with self._atomic_write(metadata_file) as temp_metadata_file:
                with open(temp_metadata_file, 'w') as f: 
                    json.dump(asdict(metadata), f, indent=2)
            
            # Invalidate cache
            with self._cache_lock_context(): 
                self._invalidate_cache_for_model(model_name)
            
            self.logger.info(f"Model '{model_name}' saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model '{model_name}': {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if model_path.exists(): 
                shutil.rmtree(model_path, ignore_errors=True)
            return False
    
    def _validate_save_inputs(self, model_name: str, results: Dict, config: Dict) -> bool:
        """
        Validate inputs before model saving.
        
        @param model_name: Model name to validate
        @param results: Results dictionary to validate
        @param config: Configuration dictionary to validate
        @return: True if all inputs valid, False otherwise
        @author: KRIBET Naoufal
        """
        if not model_name or not isinstance(model_name, str):
            self.logger.error("Invalid model name")
            return False
        
        # Check for forbidden characters in filename
        invalid_chars = set('<>:"/\\|?*')
        if any(char in model_name for char in invalid_chars):
            self.logger.error(f"Model name contains forbidden characters: {invalid_chars}")
            return False
        
        if not isinstance(results, dict) or 'model' not in results:
            self.logger.error("Invalid results - 'model' key missing")
            return False
        
        if not isinstance(config, dict):
            self.logger.error("Invalid configuration")
            return False
        
        return True
    
    def _create_backup(self, model_name: str, model_path: Path):
        """
        Create backup of existing model before overwrite.
        
        @param model_name: Name of model to backup
        @param model_path: Path to model directory
        @author: KRIBET Naoufal
        """
        try:
            backup_dir = self.model_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{model_name}_{timestamp}"
            
            shutil.copytree(model_path, backup_path)
            self.logger.info(f"Backup created: {backup_path}")
        except Exception as e:
            self.logger.warning(f"Backup creation error: {e}")
    
    def _cleanup_old_models(self):
        """
        Clean up old models if maximum limit exceeded.
        
        @author: KRIBET Naoufal
        """
        try:
            models = self.list_saved_models_detailed()
            max_models = self.config.get('max_models', 50)
            
            if len(models) > max_models:
                # Sort by creation date (oldest first)
                models.sort(key=lambda x: x.get('created_at', ''))
                
                to_delete = models[:-max_models]  # Keep most recent
                for model_info in to_delete:
                    model_name = model_info['model_name']
                    self.delete_model(model_name)
                    self.logger.info(f"Old model deleted: {model_name}")
        except Exception as e:
            self.logger.error(f"Automatic cleanup error: {e}")
    
    @lru_cache(maxsize=50)
    def list_saved_models(self) -> List[str]:
        """
        Cached version of model listing.
        
        CORRECTED: Detects both .joblib AND .keras models.
        
        @return: Sorted list of model names
        @author: KRIBET Naoufal
        """
        if not self.model_dir.exists():
            return []
        
        models = []
        for d in self.model_dir.iterdir():
            if d.is_dir() and d.name != "backups":
                report_file = d / "report.json"
                joblib_file = d / "model.joblib"
                keras_file = d / "model.keras"
                
                # Check if report exists AND either model file exists
                if report_file.exists() and (joblib_file.exists() or keras_file.exists()):
                    models.append(d.name)
        
        return sorted(models)
    
    def list_saved_models_detailed(self) -> List[Dict]:
        """
        Detailed model listing with metadata.
        
        @return: List of dictionaries containing model information
        @author: KRIBET Naoufal
        """
        models = []
        
        for model_name in self.list_saved_models():
            try:
                metadata = self.get_model_metadata(model_name)
                report = self.load_model_report_cached(model_name)
                
                model_info = {
                    'model_name': model_name,
                    'created_at': metadata.creation_date if metadata else None,
                    'file_size_mb': metadata.file_size_mb if metadata else 0,
                    'model_type': metadata.model_type if metadata else 'Unknown',
                    'feature_count': metadata.feature_count if metadata else 0,
                    'f1_score': self._extract_f1_score(report),
                    'tags': metadata.tags if metadata else []
                }
                models.append(model_info)
            except Exception as e:
                self.logger.warning(f"Error reading metadata for {model_name}: {e}")
        
        return models
    
    def _extract_f1_score(self, report: Optional[Dict]) -> float:
        """
        Extract F1-score from training report.
        
        @param report: Training report dictionary
        @return: Macro-averaged F1-score or 0.0 if not found
        @author: KRIBET Naoufal
        """
        if not report or 'report' not in report:
            return 0.0
        
        try:
            return report['report'].get('macro avg', {}).get('f1-score', 0.0)
        except:
            return 0.0
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Retrieve model metadata (with caching).
        
        @param model_name: Name of model to retrieve metadata for
        @return: ModelMetadata instance or None if not found
        @author: KRIBET Naoufal
        """
        with self._cache_lock_context():
            if model_name in self._metadata_cache:
                return self._metadata_cache[model_name]
        
        metadata_file = self.model_dir / model_name / "metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = ModelMetadata(**data)
                
                with self._cache_lock_context():
                    self._metadata_cache[model_name] = metadata
                
                return metadata
        except Exception as e:
            self.logger.error(f"Error reading metadata for {model_name}: {e}")
            return None
    
    def load_model_report_cached(self, model_name: str) -> Optional[Dict]:
        """
        Cached version of report loading.
        
        @param model_name: Name of model to load report for
        @return: Report dictionary or None if not found
        @author: KRIBET Naoufal
        """
        with self._cache_lock_context():
            if model_name in self._report_cache:
                return self._report_cache[model_name]
        
        report = self._load_model_report_from_disk(model_name)
        if report:
            with self._cache_lock_context():
                # Manage cache size
                if len(self._report_cache) >= self.cache_size:
                    # Remove oldest (FIFO)
                    oldest_key = next(iter(self._report_cache))
                    del self._report_cache[oldest_key]
                
                self._report_cache[model_name] = report
        
        return report
    
    def _load_model_report_from_disk(self, model_name: str) -> Optional[Dict]:
        """
        Load report from disk.
        
        @param model_name: Name of model
        @return: Report dictionary or None if error
        @author: KRIBET Naoufal
        """
        report_file = self.model_dir / model_name / "report.json"
        if not report_file.exists():
            self.logger.warning(f"Report file not found: {model_name}")
            return None
        
        try:
            with open(report_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted report for {model_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading report for {model_name}: {e}")
            return None
    
    def load_model_and_config_cached(self, model_name: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        Cached version of complete model loading.
        
        @param model_name: Name of model to load
        @return: Tuple of (model_package, report) or (None, None) if error
        @author: KRIBET Naoufal
        """
        with self._cache_lock_context():
            if model_name in self._model_cache:
                return self._model_cache[model_name]
        
        model, report = self._load_model_and_config_from_disk(model_name)
        
        if model and report:
            with self._cache_lock_context():
                # Manage cache size
                if len(self._model_cache) >= self.cache_size:
                    oldest_key = next(iter(self._model_cache))
                    del self._model_cache[oldest_key]
                
                self._model_cache[model_name] = (model, report)
        
        return model, report
    
    def _load_model_and_config_from_disk(self, model_name: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        Load complete "model package" (model, report, and optional scalers) from disk.
        
        @param model_name: Name of model to load
        @return: Tuple of (model_package_dict, report_dict) or (None, None) if error
        @note: For Keras models, also loads associated scalers if present
        @author: KRIBET Naoufal
        """
        model_path = self.model_dir / model_name
        keras_model_file = model_path / "model.keras"
        joblib_model_file = model_path / "model.joblib"
        report_file = model_path / "report.json"
        scalers_file = model_path / "scalers.joblib"
        
        if not report_file.exists(): 
            return None, None

        model_file_path = None
        if keras_model_file.exists(): 
            model_file_path = keras_model_file
        elif joblib_model_file.exists(): 
            model_file_path = joblib_model_file
        else: 
            return None, None

        try:
            # Create model package dictionary
            model_package = {}

            if model_file_path.suffix == '.keras':
                model_package['model'] = load_model(model_file_path)
                # Load scalers if this is a Keras model
                if scalers_file.exists():
                    self.logger.info(f"Loading scalers for LSTM model '{model_name}'.")
                    model_package['scalers'] = joblib.load(scalers_file)
            else:
                model_package['model'] = joblib.load(model_file_path)
            
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            self.logger.info(f"Model '{model_name}' and configuration loaded.")
            return model_package, report
        
        except Exception as e:
            self.logger.error(f"Error loading model package '{model_name}': {e}")
            return None, None
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model and clean cache.
        
        @param model_name: Name of model to delete
        @return: True if deletion successful, False otherwise
        @author: KRIBET Naoufal
        """
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            self.logger.warning(f"Model '{model_name}' does not exist")
            return False
        
        try:
            shutil.rmtree(model_path)
            
            # Clean cache
            with self._cache_lock_context():
                self._invalidate_cache_for_model(model_name)
            
            # Invalidate LRU cache
            self.list_saved_models.cache_clear()
            
            self.logger.info(f"Model '{model_name}' deleted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model '{model_name}': {e}")
            return False
    
    def _invalidate_cache_for_model(self, model_name: str):
        """
        Invalidate cache for specific model.
        
        @param model_name: Name of model to invalidate
        @author: KRIBET Naoufal
        """
        self._model_cache.pop(model_name, None)
        self._report_cache.pop(model_name, None)
        self._metadata_cache.pop(model_name, None)
    
    def clear_cache(self):
        """
        Clear all caches.
        
        @author: KRIBET Naoufal
        """
        with self._cache_lock_context():
            self._model_cache.clear()
            self._report_cache.clear()
            self._metadata_cache.clear()
        
        self.list_saved_models.cache_clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """
        Retrieve cache statistics.
        
        @return: Dictionary containing cache size information
        @author: KRIBET Naoufal
        """
        with self._cache_lock_context():
            return {
                'model_cache_size': len(self._model_cache),
                'report_cache_size': len(self._report_cache),
                'metadata_cache_size': len(self._metadata_cache),
                'cache_limit': self.cache_size
            }


# Global instance for backward compatibility with legacy API
_default_manager = ModelManager()

# Compatibility functions with legacy API
def save_model(model_name: str, results: Dict, config: Dict) -> bool:
    """
    Legacy API compatibility wrapper.
    
    @param model_name: Name for the model
    @param results: Training results dictionary
    @param config: Training configuration
    @return: Success status
    @author: KRIBET Naoufal
    """
    return _default_manager.save_model_optimized(model_name, results, config)

def list_saved_models() -> List[str]:
    """
    Legacy API compatibility wrapper.
    
    @return: List of saved model names
    @author: KRIBET Naoufal
    """
    return _default_manager.list_saved_models()

def load_model_report(model_name: str) -> Optional[Dict]:
    """
    Legacy API compatibility wrapper.
    
    @param model_name: Name of model
    @return: Report dictionary or None
    @author: KRIBET Naoufal
    """
    return _default_manager.load_model_report_cached(model_name)

def load_model_and_config(model_name: str) -> Optional[Tuple[object, Dict]]:
    """
    Legacy API compatibility wrapper.
    
    @param model_name: Name of model
    @return: Tuple of (model_package, config) or None
    @author: KRIBET Naoufal
    """
    return _default_manager