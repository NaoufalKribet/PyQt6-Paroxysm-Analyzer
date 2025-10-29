# core/model_manager_optimized.py - Version hautement optimisée

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
    """Niveaux de compression pour les modèles"""
    NONE = 0
    LOW = 3
    MEDIUM = 6
    HIGH = 9


@dataclass
class ModelMetadata:
    """Métadonnées enrichies pour les modèles"""
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
    """Gestionnaire de modèles optimisé avec cache, compression et validation"""
    
    def __init__(self, model_dir: str = "models", cache_size: int = 10, 
                 compression_level: CompressionLevel = CompressionLevel.MEDIUM):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.cache_size = cache_size
        self.compression_level = compression_level
        
        # Cache thread-safe
        self._model_cache = {}
        self._report_cache = {}
        self._cache_lock = threading.RLock()
        self._metadata_cache = {}
        
        # Logging
        self.logger = self._setup_logger()
        
        # Configuration
        self.config_file = self.model_dir / "manager_config.json"
        self._load_or_create_config()
    
    def _setup_logger(self) -> logging.Logger:
        """Configuration du système de logging"""
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
        """Charge ou crée la configuration du gestionnaire"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.warning(f"Erreur chargement config: {e}. Création d'une nouvelle.")
                self.config = self._create_default_config()
        else:
            self.config = self._create_default_config()
        
        self._save_config()
    
    def _create_default_config(self) -> Dict:
        """Crée une configuration par défaut"""
        return {
            "version": "2.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "compression_level": self.compression_level.value,
            "auto_cleanup": True,
            "max_models": 50,
            "backup_enabled": True
        }
    
    def _save_config(self):
        """Sauvegarde la configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde config: {e}")
    
    @contextmanager
    def _cache_lock_context(self):
        """Context manager pour le verrouillage du cache"""
        self._cache_lock.acquire()
        try:
            yield
        finally:
            self._cache_lock.release()
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcule le checksum MD5 d'un fichier"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Erreur calcul checksum: {e}")
            return ""
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Retourne la taille du fichier en MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def _extract_model_info(self, results: Dict) -> Tuple[str, int, int]:
        """Extrait les informations du modèle depuis les résultats"""
        model = results.get('model')
        model_type = type(model).__name__ if model else "Unknown"
        
        # Essayer d'extraire le nombre de features
        feature_count = 0
        if hasattr(model, 'n_features_in_'):
            feature_count = model.n_features_in_
        elif hasattr(model, 'feature_names_in_'):
            feature_count = len(model.feature_names_in_)
        
        # Essayer d'extraire le nombre d'échantillons d'entraînement
        training_samples = 0
        if 'plot_data' in results and 'y_test' in results['plot_data']:
            training_samples = len(results['plot_data']['y_test'])
        
        return model_type, feature_count, training_samples
    
    def _create_metadata(self, model_name: str, results: Dict, model_file: Path) -> ModelMetadata:
        """Crée les métadonnées du modèle"""
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
        """Écriture atomique pour éviter la corruption"""
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            yield temp_path
            # Déplacement atomique
            temp_path.replace(file_path)
        except Exception:
            # Nettoyage en cas d'erreur
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    # Dans Core/model_manager.py, REMPLACEZ cette fonction

    def save_model_optimized(self, model_name: str, results: Dict, config: Dict, 
                           tags: List[str] = None, overwrite: bool = False) -> bool:
        """
        Sauvegarde optimisée avec validation, compression et métadonnées enrichies.
        CORRIGÉ : Gère les modèles Keras, scikit-learn, et les scalers pour les modèles de prévision.
        """
        self.logger.info(f"Début sauvegarde modèle '{model_name}'")
        
        if not self._validate_save_inputs(model_name, results, config):
            return False
        
        model_path = self.model_dir / model_name
        
        if model_path.exists() and not overwrite:
            self.logger.warning(f"Modèle '{model_name}' existe déjà. Utilisez overwrite=True.")
            return False
        
        try:
            model_path.mkdir(exist_ok=True, parents=True)
            
            model_obj = results['model']
            
            if isinstance(model_obj, KerasModel):
                model_file = model_path / "model.keras"
                self.logger.info("Détection d'un modèle Keras. Utilisation de model.save().")
                with self._atomic_write(model_file) as temp_model_file:
                    model_obj.save(temp_model_file)
                
                # --- NOUVEL AJOUT : SAUVEGARDE DES SCALERS POUR LSTM ---
                if 'scalers' in results:
                    scalers_file = model_path / "scalers.joblib"
                    self.logger.info("Détection de scalers pour le modèle LSTM. Sauvegarde...")
                    joblib.dump(results['scalers'], scalers_file)
                # --- FIN DE L'AJOUT ---

            else:
                model_file = model_path / "model.joblib"
                self.logger.info("Détection d'un modèle scikit-learn. Utilisation de joblib.dump().")
                with self._atomic_write(model_file) as temp_model_file:
                    joblib.dump(model_obj, temp_model_file, compress=self.compression_level.value)

            report_file = model_path / "report.json"
            metadata_file = model_path / "metadata.json"
            
            metadata = self._create_metadata(model_name, results, model_file)
            if tags: metadata.tags.extend(tags)
            
            report_data = {
                'model_name': model_name, 'best_params': results.get('best_params', {}),
                'report': results.get('report', {}), 'training_config': config,
                'created_at': metadata.creation_date, 'model_type': metadata.model_type,
                'version': metadata.version
            }
            
            self.logger.info("Sauvegarde du rapport...")
            with self._atomic_write(report_file) as temp_report_file:
                with open(temp_report_file, 'w') as f: json.dump(report_data, f, indent=2)
            
            self.logger.info("Sauvegarde des métadonnées...")
            with self._atomic_write(metadata_file) as temp_metadata_file:
                with open(temp_metadata_file, 'w') as f: json.dump(asdict(metadata), f, indent=2)
            
            with self._cache_lock_context(): self._invalidate_cache_for_model(model_name)
            
            self.logger.info(f"Modèle '{model_name}' sauvegardé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle '{model_name}': {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if model_path.exists(): shutil.rmtree(model_path, ignore_errors=True)
            return False
    
    def _validate_save_inputs(self, model_name: str, results: Dict, config: Dict) -> bool:
        """Valide les entrées pour la sauvegarde"""
        if not model_name or not isinstance(model_name, str):
            self.logger.error("Nom de modèle invalide")
            return False
        
        # Vérifier les caractères interdits
        invalid_chars = set('<>:"/\\|?*')
        if any(char in model_name for char in invalid_chars):
            self.logger.error(f"Nom de modèle contient des caractères interdits: {invalid_chars}")
            return False
        
        if not isinstance(results, dict) or 'model' not in results:
            self.logger.error("Résultats invalides - 'model' manquant")
            return False
        
        if not isinstance(config, dict):
            self.logger.error("Configuration invalide")
            return False
        
        return True
    
    def _create_backup(self, model_name: str, model_path: Path):
        """Crée une sauvegarde du modèle"""
        try:
            backup_dir = self.model_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{model_name}_{timestamp}"
            
            shutil.copytree(model_path, backup_path)
            self.logger.info(f"Backup créé: {backup_path}")
        except Exception as e:
            self.logger.warning(f"Erreur création backup: {e}")
    
    def _cleanup_old_models(self):
        """Nettoie les anciens modèles si limite dépassée"""
        try:
            models = self.list_saved_models_detailed()
            max_models = self.config.get('max_models', 50)
            
            if len(models) > max_models:
                # Trier par date de création (plus anciens en premier)
                models.sort(key=lambda x: x.get('created_at', ''))
                
                to_delete = models[:-max_models]  # Garder les plus récents
                for model_info in to_delete:
                    model_name = model_info['model_name']
                    self.delete_model(model_name)
                    self.logger.info(f"Modèle ancien supprimé: {model_name}")
        except Exception as e:
            self.logger.error(f"Erreur nettoyage automatique: {e}")
    
    @lru_cache(maxsize=50)
    def list_saved_models(self) -> List[str]:
        """
        Version cachée de la liste des modèles.
        CORRIGÉ : Détecte les modèles .joblib ET .keras.
        """
        if not self.model_dir.exists():
            return []
        
        models = []
        for d in self.model_dir.iterdir():
            if d.is_dir() and d.name != "backups":
                report_file = d / "report.json"
                joblib_file = d / "model.joblib"
                keras_file = d / "model.keras" # On définit le chemin pour le modèle Keras
                
                # --- LA CORRECTION EST ICI ---
                # On vérifie si un rapport existe ET si L'UN OU L'AUTRE des fichiers de modèle existe.
                if report_file.exists() and (joblib_file.exists() or keras_file.exists()):
                    models.append(d.name)
        
        return sorted(models)
    
    def list_saved_models_detailed(self) -> List[Dict]:
        """Liste détaillée des modèles avec métadonnées"""
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
                self.logger.warning(f"Erreur lecture métadonnées pour {model_name}: {e}")
        
        return models
    
    def _extract_f1_score(self, report: Optional[Dict]) -> float:
        """Extrait le F1-score du rapport"""
        if not report or 'report' not in report:
            return 0.0
        
        try:
            return report['report'].get('macro avg', {}).get('f1-score', 0.0)
        except:
            return 0.0
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Récupère les métadonnées d'un modèle (avec cache)"""
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
            self.logger.error(f"Erreur lecture métadonnées {model_name}: {e}")
            return None
    
    def load_model_report_cached(self, model_name: str) -> Optional[Dict]:
        """Version cachée du chargement de rapport"""
        with self._cache_lock_context():
            if model_name in self._report_cache:
                return self._report_cache[model_name]
        
        report = self._load_model_report_from_disk(model_name)
        if report:
            with self._cache_lock_context():
                # Gérer la taille du cache
                if len(self._report_cache) >= self.cache_size:
                    # Supprimer le plus ancien (FIFO)
                    oldest_key = next(iter(self._report_cache))
                    del self._report_cache[oldest_key]
                
                self._report_cache[model_name] = report
        
        return report
    
    def _load_model_report_from_disk(self, model_name: str) -> Optional[Dict]:
        """Charge le rapport depuis le disque"""
        report_file = self.model_dir / model_name / "report.json"
        if not report_file.exists():
            self.logger.warning(f"Fichier rapport introuvable: {model_name}")
            return None
        
        try:
            with open(report_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Rapport corrompu {model_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Erreur lecture rapport {model_name}: {e}")
            return None
    
    def load_model_and_config_cached(self, model_name: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """Version cachée du chargement de modèle complet"""
        with self._cache_lock_context():
            if model_name in self._model_cache:
                return self._model_cache[model_name]
        
        model, report = self._load_model_and_config_from_disk(model_name)
        
        if model and report:
            with self._cache_lock_context():
                # Gérer la taille du cache
                if len(self._model_cache) >= self.cache_size:
                    oldest_key = next(iter(self._model_cache))
                    del self._model_cache[oldest_key]
                
                self._model_cache[model_name] = (model, report)
        
        return model, report
    
    # Dans Core/model_manager.py, REMPLACEZ cette fonction

    def _load_model_and_config_from_disk(self, model_name: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        Charge le "paquet modèle" (modèle, rapport, et scalers optionnels) depuis le disque.
        """
        model_path = self.model_dir / model_name
        keras_model_file = model_path / "model.keras"
        joblib_model_file = model_path / "model.joblib"
        report_file = model_path / "report.json"
        scalers_file = model_path / "scalers.joblib" # Chemin du fichier des scalers
        
        if not report_file.exists(): return None, None

        model_file_path = None
        if keras_model_file.exists(): model_file_path = keras_model_file
        elif joblib_model_file.exists(): model_file_path = joblib_model_file
        else: return None, None

        try:
            # --- MODIFICATION POUR RETOURNER UN PAQUET COMPLET ---
            model_package = {}

            if model_file_path.suffix == '.keras':
                model_package['model'] = load_model(model_file_path)
                # Si c'est un Keras, on cherche les scalers
                if scalers_file.exists():
                    self.logger.info(f"Chargement des scalers pour le modèle LSTM '{model_name}'.")
                    model_package['scalers'] = joblib.load(scalers_file)
            else:
                model_package['model'] = joblib.load(model_file_path)
            
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            self.logger.info(f"Modèle '{model_name}' et sa configuration chargés.")
            # On retourne le paquet et le rapport
            return model_package, report
        
        except Exception as e:
            self.logger.error(f"Erreur chargement paquet modèle '{model_name}': {e}")
            return None, None
    
    def delete_model(self, model_name: str) -> bool:
        """Supprime un modèle et nettoie le cache"""
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            self.logger.warning(f"Modèle '{model_name}' n'existe pas")
            return False
        
        try:
            shutil.rmtree(model_path)
            
            # Nettoyer le cache
            with self._cache_lock_context():
                self._invalidate_cache_for_model(model_name)
            
            # Invalider le cache LRU
            self.list_saved_models.cache_clear()
            
            self.logger.info(f"Modèle '{model_name}' supprimé")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur suppression modèle '{model_name}': {e}")
            return False
    
    def _invalidate_cache_for_model(self, model_name: str):
        """Invalide le cache pour un modèle spécifique"""
        self._model_cache.pop(model_name, None)
        self._report_cache.pop(model_name, None)
        self._metadata_cache.pop(model_name, None)
    
    def clear_cache(self):
        """Vide tout le cache"""
        with self._cache_lock_context():
            self._model_cache.clear()
            self._report_cache.clear()
            self._metadata_cache.clear()
        
        self.list_saved_models.cache_clear()
        self.logger.info("Cache vidé")
    
    def get_cache_stats(self) -> Dict:
        """Statistiques du cache"""
        with self._cache_lock_context():
            return {
                'model_cache_size': len(self._model_cache),
                'report_cache_size': len(self._report_cache),
                'metadata_cache_size': len(self._metadata_cache),
                'cache_limit': self.cache_size
            }


# Instance globale pour compatibilité avec l'ancienne API
_default_manager = ModelManager()

# Fonctions de compatibilité avec l'ancienne API
def save_model(model_name: str, results: Dict, config: Dict) -> bool:
    """Interface de compatibilité"""
    return _default_manager.save_model_optimized(model_name, results, config)

def list_saved_models() -> List[str]:
    """Interface de compatibilité"""
    return _default_manager.list_saved_models()

def load_model_report(model_name: str) -> Optional[Dict]:
    """Interface de compatibilité"""
    return _default_manager.load_model_report_cached(model_name)

def load_model_and_config(model_name: str) -> Optional[Tuple[object, Dict]]:
    """Interface de compatibilité"""
    return _default_manager.load_model_and_config_cached(model_name)