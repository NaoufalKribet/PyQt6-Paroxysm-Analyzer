# app_controller_optimized.py - Version optimisée

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, QMutex
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional, Dict, Any, Callable
from functools import lru_cache, wraps
from contextlib import contextmanager
import logging
from ui_dialogs import ProcessingDialog

# Import des modules de votre application
from Core.data_processor import load_data, define_internal_event_cycle, balance_dataset, extract_activity_blocks
from Core.feature_extractor import extract_features
# Dans Core.model_trainer
from Core.model_trainer import (
    train_and_evaluate_knn,
    train_and_evaluate_rf,
    train_and_evaluate_lgbm,
    train_and_evaluate_nn,
    #train_and_evaluate_lstm_pinn,  # <-- AJOUTEZ CETTE LIGNE
    ModelConfig
)
from tensorflow.keras.models import Model as KerasModel
from Core.explainer import ModelExplainer
from Core.model_manager import save_model, list_saved_models, load_model_report, load_model_and_config
from Core.data_processor import create_binary_target
from Core.synthetic_data_generator import generate_realistic_synthetic_data
# Dans app_controller.py, en haut avec les autres imports
from Core.model_trainer import create_sequences, derive_state_from_forecast



class WorkerThread(QThread):
    """
    Thread worker optimisé pour les opérations longues.
    Il peut désormais signaler sa progression et être annulé depuis l'extérieur.
    """
    result_ready = pyqtSignal(object)
    
    error_occurred = pyqtSignal(str, str)

    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, func: Callable, *args, **kwargs):
        """
        Initialise le worker.
        :param func: La fonction à exécuter dans le thread.
        :param args: Les arguments positionnels pour la fonction.
        :param kwargs: Les arguments par mot-clé pour la fonction.
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
        # NOUVEAU : Flag interne pour gérer la demande d'annulation.
        # Il est volatile pour assurer une visibilité correcte entre les threads.
        self._is_cancelled = False

    def run(self):
        """
        Le cœur du thread. Cette méthode est exécutée lorsque le thread est démarré.
        Elle prépare et appelle la fonction cible.
        """
        try:
            # --- Injection des callbacks de communication ---
            # Nous ajoutons dynamiquement deux arguments à la fonction cible :
            # 1. 'update_progress': un callback pour signaler la progression.
            # 2. 'is_cancelled': une fonction pour vérifier si l'annulation a été demandée.
            self.kwargs['update_progress'] = self.progress_updated.emit
            self.kwargs['is_cancelled'] = lambda: self._is_cancelled
            
            # Appel de la fonction cible avec tous ses arguments
            result = self.func(*self.args, **self.kwargs)
            
            # Si le flag d'annulation a été activé pendant l'exécution,
            # on ne signale pas que le résultat est prêt, même si la fonction
            # a retourné quelque chose. Le thread se terminera silencieusement.
            if self._is_cancelled:
                return
            else:
                self.result_ready.emit(result)

        except Exception as e:
            # En cas d'exception, on ne l'émet comme une erreur que si la tâche
            # n'a pas été annulée manuellement.
            if not self._is_cancelled:
                import traceback
                error_message = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                self.error_occurred.emit(f"Erreur dans le Worker ({type(e).__name__})", error_message)

    def cancel(self):
        """
        Méthode publique appelée depuis le thread principal (UI) pour demander
        l'arrêt de la tâche.
        """
        self.progress_updated.emit(self.progress_bar.value(), "Annulation demandée...")
        self._is_cancelled = True
class DataCache:
    """Cache intelligent pour éviter les recalculs"""
    def __init__(self, max_size: int = 5):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        self.mutex = QMutex()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock():
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        with self._lock():
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    @contextmanager
    def _lock(self):
        self.mutex.lock()
        try:
            yield
        finally:
            self.mutex.unlock()


class AppConfig:
    """Configuration centralisée"""
    RANDOM_STATE = 42
    DEFAULT_DECAY_RATE = -0.1
    DEFAULT_SIMULATION_SPEED = 100
    CACHE_SIZE = 10
    
    # Ratios par défaut
    DEFAULT_PRE_EVENT_RATIO = 1.0
    DEFAULT_POST_EVENT_RATIO = 0.5


class AppController(QObject):
    # --- SIGNAUX (inchangés) ---
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    data_loaded = pyqtSignal(pd.DataFrame)
    overview_ready = pyqtSignal(pd.DataFrame)
    feature_extraction_finished = pyqtSignal(pd.DataFrame)
    training_finished = pyqtSignal(dict)
    model_list_updated = pyqtSignal(list)
    comparison_ready = pyqtSignal(dict, dict)
    leaderboard_ready = pyqtSignal(list)
    external_data_loaded = pyqtSignal(str)
    external_prediction_finished = pyqtSignal(pd.DataFrame)
    simulation_started = pyqtSignal(dict)
    simulation_step_updated = pyqtSignal(dict)
    simulation_finished = pyqtSignal(str)
    explanation_ready = pyqtSignal(object, object, object, list) # (model, shap_values, base_value, instance_data)
    segmentation_finished = pyqtSignal(dict)
    segmentation_log_updated = pyqtSignal(str)
    whatif_analysis_finished = pyqtSignal(dict)


    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.cache = DataCache(self.config.CACHE_SIZE)
        self.logger = self._setup_logger()
        
        # État de l'application (optimisé)
        self._reset_state()
        self.current_explainer = None # Ajouter cet attribut
        
        # Simulation
        self._setup_simulation()
        
        # Thread management
        self.active_workers = []

    def _setup_logger(self) -> logging.Logger:
        """Configuration du logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def run_whatif_analysis(self, model_name, scenario_df):
        if not model_name or scenario_df is None:
            self.error_occurred.emit("Erreur", "Veuillez sélectionner un modèle et avoir des données de scénario.")
            return

        worker = self._run_in_thread(
            self._whatif_task,
            self.whatif_analysis_finished.emit,
            model_name,
            scenario_df
        )
        return worker

    def _whatif_task(self, model_name: str, scenario_df: pd.DataFrame, 
                     update_progress: Callable, is_cancelled: Callable):
        """Tâche de fond pour l'analyse de scénario."""
        update_progress(10, f"Chargement du modèle '{model_name}'...")
        model, config = load_model_and_config(model_name)
        if not model or not config: raise ValueError("Impossible de charger le modèle.")
        
        update_progress(30, "Recalcul des features sur le scénario modifié...")
        feature_config = config['training_config']['feature_config']
        scenario_features = extract_features(scenario_df, feature_config)
        
        update_progress(70, "Prédiction sur les nouvelles features...")
        model_input = scenario_features.reindex(columns=model.feature_names_in_, fill_value=0)
        probabilities = model.predict_proba(model_input)
        
        update_progress(100, "Terminé.")
        return {'probabilities': probabilities, 'dates': scenario_df['Date'].values, 'class_names': model.classes_}

    def _reset_state(self):
        """Réinitialise l'état de l'application"""
        self.original_df = None
        #self.processed_df = None
        self.feature_matrix = None
        self.last_training_results = None
        self.last_feature_config = None
        self.external_df = None
        
    def _setup_simulation(self):
        """Initialise les composants de simulation"""
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self._simulation_step)
        self.simulation_data = None
        self.simulation_feature_matrix = None
        self.simulation_model = None
        self.simulation_model_config = None
        self.simulation_state = "STOPPED"
        self.simulation_index = 0
        self.simulation_speed_ms = self.config.DEFAULT_SIMULATION_SPEED

    # --- DÉCORATEURS UTILITAIRES ---
    def _require_data(func):
        """Décorateur pour vérifier la présence de données"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.original_df is None:
                self.error_occurred.emit("Erreur", "Veuillez d'abord charger des données.")
                return None
            return func(self, *args, **kwargs)
        return wrapper

    def _require_features(func):
        """Décorateur pour vérifier la présence de features"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.feature_matrix is None:
                self.error_occurred.emit("Erreur", "Veuillez d'abord extraire les caractéristiques.")
                return None
            return func(self, *args, **kwargs)
        return wrapper

    def _run_in_thread(self, func: Callable, callback: Optional[Callable] = None, *args, **kwargs) -> WorkerThread:
        # Cette fonction DOIT maintenant retourner le worker
        worker = WorkerThread(func, *args, **kwargs)
        if callback:
            worker.result_ready.connect(callback)
        worker.error_occurred.connect(self.error_occurred.emit)
        self.active_workers.append(worker)
        worker.finished.connect(lambda: self.active_workers.remove(worker))
        worker.start()
        return worker

    # --- MÉTHODES UTILITAIRES OPTIMISÉES ---
    @lru_cache(maxsize=10)
    def _create_temporal_weights(self, length: int, decay_rate: float) -> np.ndarray:
        """Version optimisée avec cache des poids temporels"""
        time_indices = np.arange(length)
        weights = np.exp(decay_rate * (time_indices - time_indices.max()))
        return weights

    def _get_cached_features(self, df_hash: str, config: dict) -> Optional[pd.DataFrame]:
        """Récupère les features du cache si disponibles"""
        cache_key = f"features_{df_hash}_{hash(str(config))}"
        return self.cache.get(cache_key)

    def _cache_features(self, df_hash: str, config: dict, features: pd.DataFrame):
        """Met en cache les features calculées"""
        cache_key = f"features_{df_hash}_{hash(str(config))}"
        self.cache.set(cache_key, features)

    # --- 1. GESTION DES DONNÉES (optimisée) ---
    def load_data_from_file(self, filepath: str) -> WorkerThread:
        """Charge les données de manière asynchrone et retourne le worker."""
        
        # Tâche pour le thread
        def _load_task(update_progress: Callable, is_cancelled: Callable):
            update_progress(10, "Ouverture du fichier...")
            df = load_data(filepath)
            if is_cancelled() or df is None: return None
            update_progress(100, "Fichier chargé.")
            return df
        
        # Callback qui s'exécute à la fin
        def _on_data_loaded(df: pd.DataFrame):
            self.original_df = df
            self.status_updated.emit(f"Fichier chargé. {len(df)} lignes.")
            self.data_loaded.emit(df)
            self.overview_ready.emit(df)
        
        # Appel unique et correct à la fin
        return self._run_in_thread(_load_task, _on_data_loaded)


    @_require_data
    def run_segmentation_training(self, params: dict):
        """
        Lance la pipeline complète pour entraîner le Segmenteur.
        """
        if not self.last_training_results or 'model' not in self.last_training_results:
            self.error_occurred.emit("Erreur", "Veuillez d'abord entraîner un modèle Détecteur (Étape 3).")
            return None
        
        # Le worker exécutera cette nouvelle tâche
        worker = self._run_in_thread(
            self._segmentation_task,
            self._on_segmentation_complete,
            params
        )
        return worker

    def _segmentation_task(self, params: dict, update_progress: Callable, is_cancelled: Callable) -> Optional[dict]:
        """
        Tâche de fond pour l'entraînement du Segmenteur.
        """
        try:
            # --- ÉTAPE 2.1 : OBTENIR LE DATASET DU SEGMENTEUR ---
            update_progress(10, "Application du Détecteur pour trouver les zones actives...")
            self.segmentation_log_updated.emit("1. Utilisation du Détecteur binaire pour prédire les zones d'activité...")
            
            detector_model = self.last_training_results['model']
            # On utilise toutes les features calculées précédemment sur l'ensemble des données
            all_features = self.feature_matrix.reindex(columns=detector_model.feature_names_in_, fill_value=0)
            detector_predictions = detector_model.predict(all_features)
            
            self.segmentation_log_updated.emit(f"   -> {sum(detector_predictions == 'Actif')} points prédits comme 'Actif'.")
            if is_cancelled(): return None

            update_progress(25, "Extraction des blocs actifs du dataset original...")
            self.segmentation_log_updated.emit("2. Extraction des segments de données correspondant à ces zones...")
            
            # Nouvelle fonction pour isoler les données
            segmenter_raw_df = extract_activity_blocks(self.original_df, pd.Series(detector_predictions, index=all_features.index))
            
            if segmenter_raw_df.empty:
                self.error_occurred.emit("Info", "Le Détecteur n'a trouvé aucune zone 'Actif'. Impossible d'entraîner le Segmenteur.")
                return None
            
            self.segmentation_log_updated.emit(f"   -> Le nouveau jeu de données pour le Segmenteur contient {len(segmenter_raw_df)} lignes.")
            if is_cancelled(): return None

            # --- ÉTAPE 2.2 : INGÉNIERIE DES CARACTÉRISTIQUES SPÉCIFIQUE ---
            update_progress(40, "Recalcul des features sur les données actives...")
            self.segmentation_log_updated.emit("3. Recalcul des caractéristiques spécifiques au régime actif...")
            
            # On utilise la même configuration de features que pour le Détecteur
            feature_config = self.last_feature_config
            segmenter_features = extract_features(segmenter_raw_df, feature_config)
            
            self.segmentation_log_updated.emit(f"   -> {segmenter_features.shape[1]} caractéristiques recalculées.")
            if is_cancelled() or segmenter_features.empty: return None

            # --- ÉTAPE 2.3 : ENTRAÎNEMENT DU SEGMENTEUR ---
            update_progress(60, "Préparation des labels pour la segmentation...")
            self.segmentation_log_updated.emit("4. Préparation des labels cibles (Pre-Event, High-Paroxysm, Post-Event)...")
            
            # Préparation des labels multi-classes sur le jeu de données du segmenteur
            segmenter_labeled_df = define_internal_event_cycle(segmenter_raw_df, pre_event_ratio=0.3)

            # Alignement
            y_s = segmenter_labeled_df.set_index('Date')['Ramp']
            common_index = segmenter_features.index.intersection(y_s.index)
            X_s = segmenter_features.loc[common_index]
            y_s = y_s.loc[common_index]

            if is_cancelled(): return None

            update_progress(75, "Entraînement du modèle de segmentation...")
            self.segmentation_log_updated.emit("5. Lancement de l'entraînement du Segmenteur multi-classes...")
            
            # Séparation temporelle des données du Segmenteur
            n = len(X_s)
            train_end = int(n * 0.8)
            X_s_train, y_s_train = X_s.iloc[:train_end], y_s.iloc[:train_end]
            X_s_test, y_s_test = X_s.iloc[train_end:], y_s.iloc[train_end:]

            # Utilisation des fonctions d'entraînement existantes
            # Note: Le Segmenteur pourrait bénéficier de son propre ModelConfig
            model_config = ModelConfig(**params.get('model_params', {}))
            seg_results = train_and_evaluate_rf(X_s_train, y_s_train, None, None, X_s_test, y_s_test, model_config)
            
            self.segmentation_log_updated.emit("   -> Entraînement terminé.")
            update_progress(100, "Terminé.")

            # Ajouter les données de plot
            seg_results['plot_data'] = {'y_test': y_s_test, 'predictions': seg_results['predictions']}
            return seg_results

        except Exception as e:
            import traceback
            self.logger.error(f"Erreur dans la tâche de segmentation: {e}\n{traceback.format_exc()}")
            raise e

    def _on_segmentation_complete(self, results: dict):
        """Callback pour gérer les résultats du Segmenteur."""
        if results:
            self.status_updated.emit("Entraînement du Segmenteur terminé avec succès.")
            self.segmentation_finished.emit(results)
    @_require_data
    def run_balancing_preview(self):
        """Génère un aperçu des données équilibrées"""
        def _balance_data():
            return balance_dataset(self.original_df.copy(), params={})
        
        def _on_balanced(balanced_df):
            if balanced_df is not None:
                self.data_loaded.emit(balanced_df)
                self.status_updated.emit("Aperçu des données équilibrées généré.")
            else:
                self.error_occurred.emit("Erreur", "L'équilibrage a échoué.")
        
        _ = self._run_in_thread(_balance_data, _on_balanced)

    # --- 2. EXTRACTION DE CARACTÉRISTIQUES (optimisée) ---
    @_require_data
    def run_feature_extraction(self, config: dict) -> Optional[WorkerThread]:
        """
        Extraction optimisée avec cache, utilisant toujours les données originales
        et retournant un worker pour le suivi de la progression.
        """
        df_hash = str(hash(self.original_df.values.tobytes()))
        
        # 1. Vérifier le cache d'abord
        cached_features = self._get_cached_features(df_hash, config)
        if cached_features is not None:
            self.feature_matrix = cached_features
            self.last_feature_config = config
            self.status_updated.emit(f"Features récupérées du cache. {cached_features.shape[1]} caractéristiques.")
            self.feature_extraction_finished.emit(cached_features)
            # Pas de tâche longue, on retourne None pour que l'UI n'affiche pas de dialogue
            return None

        # 2. Si pas de cache, préparer la tâche qui sera exécutée dans le thread
        def _extract_task(update_progress: Callable, is_cancelled: Callable):
            """Cette fonction locale accepte maintenant les callbacks."""
            # On passe les callbacks à la fonction d'extraction du module feature_extractor
            features = extract_features(
                self.original_df.copy(), 
                config,
                update_progress=update_progress,
                is_cancelled=is_cancelled
            )
            # La fonction extract_features retournera None si elle est annulée
            return features
        
        # 3. Définir le callback qui s'exécutera à la fin de la tâche
        def _on_features_extracted(features):
            # Vérifier que la tâche n'a pas été annulée (auquel cas features est None)
            if features is not None:
                self.feature_matrix = features
                self.last_feature_config = config
                self._cache_features(df_hash, config, features)
                self.status_updated.emit(f"Extraction terminée. {features.shape[1]} caractéristiques générées.")
                self.feature_extraction_finished.emit(features)
        
        # 4. Lancer le thread avec la bonne tâche et le bon callback, puis retourner le worker
        return self._run_in_thread(_extract_task, _on_features_extracted)
        
    

    # --- 3. ENTRAÎNEMENT OPTIMISÉ (divisé en méthodes plus petites) ---
    @_require_data # On n'a besoin que des données brutes au départ
    def run_full_training(self, params: dict):
        """Lance l'entraînement de manière asynchrone avec un pipeline intégré."""
        
        # Le worker exécutera cette nouvelle fonction _integrated_training_task
        worker = self._run_in_thread(
            self._integrated_training_task, 
            self._on_training_complete, 
            params
        )
        return worker

    def _integrated_training_task(self, params: dict, update_progress: Callable, is_cancelled: Callable) -> Optional[dict]:
        """
        Tâche d'entraînement complète et intégrée, avec un cloisonnement strict
        pour éviter la fuite de données.
        """
        try:
            update_progress(5, "Étape 1/4 : Préparation des labels cibles...")
            # 1. On prépare le DataFrame avec les labels finaux ('Calm', 'Pre-Event', etc.).
            df_labeled = self._prepare_target_data(params)
            if is_cancelled(): return None

            # --- C'EST ICI LA CORRECTION FONDAMENTALE ---
            # 2. On crée une version du DataFrame SANS les labels pour l'extraction.
            cols_for_feature_gen = ['Date', 'VRP']
            df_for_features = df_labeled[cols_for_feature_gen].copy()
            update_progress(15, "Étape 2/4 : Extraction des caractéristiques (mode aveugle)...")
            
            # 3. On passe ce DataFrame "aveugle" à l'extracteur. Votre fonction
            #    _process_single_segment (qui est déjà correcte) est appelée ici.
            feature_config = params['feature_config']
            feature_matrix = extract_features(
                df_for_features, # <--- ON UTILISE LE DATAFRAME SANS LABELS
                feature_config,
                update_progress=lambda v, m: update_progress(15 + int(v * 0.3), f"Étape 2/4 : {m}"),
                is_cancelled=is_cancelled
            )
            # --- FIN DE LA PARTIE CRITIQUE ---

            if feature_matrix is None: return None
            
            self.feature_matrix = feature_matrix

            update_progress(50, "Étape 3/4 : Alignement et partitionnement des données...")
            # 4. On réunit maintenant les caractéristiques "honnêtes" et les labels.
            labels_series = df_labeled.set_index('Date')['Ramp']
            common_index = feature_matrix.index.intersection(labels_series.index)
            
            X = feature_matrix.loc[common_index]
            y = labels_series.loc[common_index]
            if is_cancelled(): return None

            # La suite du processus...
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y, params)
            y_train_val = pd.concat([y_train, y_val]) if y_val is not None else y_train
            sample_weights = self._compute_sample_weights(y_train_val, params)
            
            update_progress(70, "Étape 4/4 : Entraînement du modèle...")
            results = self._execute_training(X_train, y_train, X_val, y_val, X_test, y_test, params, sample_weights)
            if is_cancelled(): return None

            update_progress(95, "Préparation des données pour l'affichage...")
            results['plot_data'] = self._prepare_plot_data(X_train, X_val, X_test, y_test)
            
            update_progress(100, "Terminé.")
            return results

        except Exception as e:
            import traceback
            self.logger.error(f"Erreur dans le pipeline d'entraînement: {e}\n{traceback.format_exc()}")
            raise e

    def _train_model(self, params: dict, update_progress: Callable, is_cancelled: Callable) -> Optional[dict]:
        """
        Logique d'entraînement séparée pour threading, qui accepte les callbacks de
        progression et d'annulation.
        """
        try:
            update_progress(5, "Préparation de l'environnement d'entraînement...")
            df_with_target = self._prepare_target_data(params)
            if is_cancelled(): return None

            update_progress(15, "Alignement des caractéristiques et des labels...")
            X, y = self._align_features_and_labels(df_with_target)
            if is_cancelled(): return None

            update_progress(20, "Partitionnement temporel des données...")
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y, params)
            if is_cancelled(): return None

            # Normalisation si demandée
            if params.get('normalize', False):
                update_progress(25, "Normalisation des données...")
                scaler = StandardScaler().fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
                if not X_val.empty:
                    X_val = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
                if not X_test.empty:
                    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            if is_cancelled(): return None

            update_progress(30, "Calcul des poids temporels...")
            y_train_val = pd.concat([y_train, y_val])
            sample_weights = self._compute_sample_weights(y_train_val, params)
            if is_cancelled(): return None
 
            update_progress(30, f"Lancement de l'entraînement ({params['model_type']})...")
            results = self._execute_training(X_train, y_train, X_val, y_val, X_test, y_test, params, sample_weights)
            if is_cancelled(): return None

            update_progress(95, "Préparation des données pour l'affichage...")
            results['plot_data'] = self._prepare_plot_data(X_train, X_val, X_test, y_test)
            
            update_progress(100, "Terminé.")
            return results
        except Exception as e:
            # Remonter l'erreur pour qu'elle soit gérée par le worker
            raise e

    def _prepare_target_data(self, params: dict) -> pd.DataFrame:
        """
        Prépare les données avec la colonne cible finale.
        CORRIGÉ POUR UTILISER 'Actif' POUR LE MODE BINAIRE.
        """
        df_with_target = self.original_df.copy()

        # Le param 'use_binary_mode' viendra de notre checkbox dans l'UI
        if params.get('use_binary_mode', False):
            self.status_updated.emit("Création d'une cible binaire ('Calm' vs 'Actif')...")
            # On appelle la fonction qui transforme 'High'/'Low' en 'Calm'/'Actif'
            return create_binary_target(df_with_target, positive_class_name='Actif')

        elif params.get('use_cycle_target', False):
            self.status_updated.emit("Création d'une cible multi-classes (4 états)...")
            return define_internal_event_cycle(
                df_with_target, 
                pre_event_ratio=params.get('pre_event_ratio', 0.3),
                post_event_ratio=params.get('post_event_ratio', 0.2)
            )
        else:
            # Fallback si aucune option n'est cochée : utiliser les données telles quelles
            return df_with_target

    def _align_features_and_labels(self, df_with_target: pd.DataFrame) -> tuple:
        """Aligne les features et les labels"""
        self.status_updated.emit("Alignement des caractéristiques et des labels...")
        labels_df = df_with_target.set_index('Date')
        common_index = self.feature_matrix.index.intersection(labels_df.index)
        
        if common_index.empty:
            raise ValueError("Aucune donnée commune trouvée entre features et labels.")
        
        X = self.feature_matrix.loc[common_index]
        y = labels_df.loc[common_index]['Ramp']
        return X, y

    def _split_data(self, X: pd.DataFrame, y: pd.Series, params: dict) -> tuple:
        """Divise les données en ensembles d'entraînement et de test"""
        self.status_updated.emit("Partitionnement temporel des données...")
        n = len(X)
        train_end_idx = int(n * params['train_ratio'] / 100)
        val_end_idx = train_end_idx + int(n * params['val_ratio'] / 100)

        X_train, y_train = X.iloc[:train_end_idx], y.iloc[:train_end_idx]
        X_val, y_val = X.iloc[train_end_idx:val_end_idx], y.iloc[train_end_idx:val_end_idx]
        X_test, y_test = X.iloc[val_end_idx:], y.iloc[val_end_idx:]
        
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _normalize_data(self, X_train_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Normalise les données"""
        self.status_updated.emit("Normalisation des données...")
        scaler = StandardScaler().fit(X_train_val)
        
        X_train_val_norm = pd.DataFrame(
            scaler.transform(X_train_val), 
            index=X_train_val.index, 
            columns=X_train_val.columns
        )
        X_test_norm = pd.DataFrame(
            scaler.transform(X_test), 
            index=X_test.index, 
            columns=X_test.columns
        )
        
        return X_train_val_norm, X_test_norm

    def _compute_sample_weights(self, y: pd.Series, params: dict) -> Optional[np.ndarray]:
        """Calcule les poids d'échantillon si nécessaire"""
        if not params.get('use_sample_weight', False):
            return None
        
        decay_rate = params.get('decay_rate', self.config.DEFAULT_DECAY_RATE)
        return self._create_temporal_weights(len(y), decay_rate)

    def _execute_training(self, X_train, y_train, X_val, y_val, X_test, y_test, params, sample_weights):
        """Exécute l'entraînement proprement dit"""
        self.status_updated.emit(f"Lancement de l'entraînement pour : {params['model_type']}...")
        
        model_config = ModelConfig(
            random_state=self.config.RANDOM_STATE,
            max_k=params['model_params']['max_k'],
            weights=params['model_params']['weights'],
            metrics=params['model_params']['metrics'],
            max_trees=params['model_params']['max_trees'],
            max_depth=params['model_params']['max_depth'],
            min_leaf=params['model_params']['min_leaf'],
            use_class_weight=params['model_params']['use_class_weight'],
            nn_epochs=params['model_params']['nn_epochs'],
            nn_batch_size=params['model_params']['nn_batch_size'],
            nn_learning_rate=params['model_params']['nn_learning_rate']
        )
        
        # Dans la méthode _execute_training de AppController
        trainer_map = {
            "Neural Network": train_and_evaluate_nn, # <-- AJOUTEZ CETTE LIGNE
            "K-Nearest Neighbors (KNN)": train_and_evaluate_knn,
            "Random Forest": train_and_evaluate_rf,
            "LightGBM": train_and_evaluate_lgbm,
            #"LSTM (Physics-Guided)": train_and_evaluate_lstm_pinn, # <-- AJOUTEZ CETTE LIGNE

        }
        
        trainer_func = trainer_map[params['model_type']]
        
        # CORRECTION : Appel de la fonction d'entraînement en passant les 6 jeux de données séparés
        results = trainer_func(
            X_train, y_train, X_val, y_val, X_test, y_test, 
            model_config, sample_weight=sample_weights
        )
        
        # Les données de plot sont gérées dans _train_model, on retourne juste les résultats ici
        return results

    def _prepare_plot_data(self, X_train, X_val, X_test, y_test):
        """Prépare les données pour les graphiques"""
        return {
            'train_df': self.original_df[self.original_df['Date'].isin(X_train.index)],
            'val_df': self.original_df[self.original_df['Date'].isin(X_val.index)], # <-- LA CLÉ MANQUANTE EST AJOUTÉE ICI
            'test_df': self.original_df[self.original_df['Date'].isin(X_test.index)],
            'y_test': y_test
        }


    # Dans app_controller.py, REMPLACEZ la méthode _on_training_complete

    def _on_training_complete(self, results: dict):
        """Callback appelé à la fin de l'entraînement."""
        self.last_training_results = results
        
        model = results.get('model')
        plot_data = results.get('plot_data')
        
        # --- CORRECTION DE L'INITIALISATION DE L'EXPLAINER ---
        if model and plot_data and not plot_data['train_df'].empty:
            # On ne crée un explainer que pour les modèles compatibles (non-Keras)
            if not isinstance(model, KerasModel):
                X_train = self.feature_matrix.loc[plot_data['train_df']['Date']]
                # On récupère le jeu de test X à partir de l'index de y_test
                X_test_shap = self.feature_matrix.loc[plot_data['y_test'].index]
                background_sample = X_train.sample(min(100, len(X_train)), random_state=self.config.RANDOM_STATE)
                
                # On passe maintenant X_test_shap comme troisième argument
                self.current_explainer = ModelExplainer(model, background_sample, X_test_shap)
            else:
                self.current_explainer = None
        else:
            self.current_explainer = None
        # --- FIN DE LA CORRECTION ---
            
        self.status_updated.emit("Entraînement terminé avec succès.")
        self.training_finished.emit(results)
    def explain_prediction(self, data_index: int):
        """
        Déclenche l'explication pour une instance de données spécifique.
        """
        if self.current_explainer is None:
            self.error_occurred.emit("Erreur", "Aucun explainer n'est chargé. Le modèle actuel n'est peut-être pas compatible.")
            return

        try:
            # --- CORRECTION : APPEL À LA NOUVELLE MÉTHODE ---
            # On appelle la méthode qui utilise les valeurs pré-calculées
            shap_values, base_value, instance_data = self.current_explainer.explain_instance_by_index(data_index)
            # --- FIN DE LA CORRECTION ---
            
            class_names = self.current_explainer.model.classes_.tolist()
            self.explanation_ready.emit(shap_values, base_value, instance_data, class_names)

        except Exception as e:
            self.error_occurred.emit("Erreur d'Explication", f"Impossible de générer l'explication SHAP : {e}")
# --- 4. GESTION/COMPARAISON DE MODÈLES (MÉTHODES MANQUANTES) ---
    def refresh_model_lists(self):
        """Met à jour les listes de modèles dans l'UI."""
        try:
            self.model_list_updated.emit(list_saved_models())
        except Exception as e:
            self.error_occurred.emit("Erreur", f"Impossible de lister les modèles sauvegardés: {e}")

    def run_comparison(self, model_a_name: str, model_b_name: str):
        """Compare deux modèles sauvegardés."""
        if not model_a_name or not model_b_name:
            self.error_occurred.emit("Avertissement", "Veuillez sélectionner deux modèles.")
            return
        try:
            # Cette opération est rapide, pas besoin de thread
            report_a = load_model_report(model_a_name)
            report_b = load_model_report(model_b_name)
            if report_a and report_b:
                self.comparison_ready.emit(report_a, report_b)
            else:
                self.error_occurred.emit("Erreur", "Impossible de charger un ou plusieurs rapports de modèle.")
        except Exception as e:
            self.error_occurred.emit("Erreur de Comparaison", str(e))

    def show_leaderboard(self):
        """Affiche le classement des modèles basé sur le F1-score macro."""
        try:
            # Cette opération est aussi très rapide
            reports = [load_model_report(name) for name in list_saved_models() if load_model_report(name)]
            sorted_reports = sorted(reports, key=lambda r: r['report'].get('macro avg', {}).get('f1-score', 0), reverse=True)
            self.leaderboard_ready.emit(sorted_reports)
        except Exception as e:
            self.error_occurred.emit("Erreur Leaderboard", str(e))


    # --- 5. PRÉDICTION EXTERNE (MÉTHODES MANQUANTES) ---
    def load_external_data(self, filepath: str) -> WorkerThread:
        """Charge un jeu de données externe et retourne le worker."""

        # Tâche pour le thread
        def _load_external_task(update_progress: Callable, is_cancelled: Callable):
            update_progress(10, "Ouverture du fichier externe...")
            df = load_data(filepath)
            if is_cancelled() or df is None: return None
            update_progress(100, "Fichier chargé.")
            return df, filepath.split('/')[-1]

        # Callback qui s'exécute à la fin
        def _on_loaded(result):
            df, filename = result
            self.external_df = df
            self.external_data_loaded.emit(f"Fichier externe chargé : {filename}")

        # Appel unique et correct à la fin
        return self._run_in_thread(_load_external_task, _on_loaded)

    # Dans app_controller.py
    # Dans app_controller.py
        # Dans app_controller.py, REMPLACEZ la fonction run_external_prediction existante

    def run_external_prediction(self, model_name: str, replicate_test_set: bool, test_set_percentage: int):
        """
        Lance la prédiction externe.
        VERSION MISE À JOUR : Gère les classifieurs et les régresseurs LSTM (PINN).
        """
        if self.external_df is None and not replicate_test_set:
            self.error_occurred.emit("Erreur", "Veuillez charger des données externes ou cocher 'Répliquer un Set de Test'.")
            return None
        if not model_name:
            self.error_occurred.emit("Erreur", "Veuillez sélectionner un modèle.")
            return None

        def _predict(update_progress: Callable, is_cancelled: Callable) -> Optional[pd.DataFrame]:
            update_progress(10, f"Chargement du paquet modèle '{model_name}'...")
            
            # load_model_and_config retourne maintenant un "paquet"
            model_package, config = load_model_and_config(model_name)
            if not model_package or not config:
                raise ValueError("Impossible de charger le paquet modèle ou sa configuration.")
            
            model = model_package['model']
            
            # Déterminer les données à utiliser
            df_for_prediction = self.original_df if replicate_test_set else self.external_df
            if replicate_test_set:
                num_rows = len(df_for_prediction)
                cutoff_index = int(num_rows * (1 - test_set_percentage / 100))
                df_for_prediction = df_for_prediction.iloc[cutoff_index:].copy()
            
            prediction_series = None

            # --- DÉCISION : S'agit-il d'un classifieur ou de notre LSTM ? ---
            if 'scalers' in model_package:
                # --- CHEMIN B : LOGIQUE POUR LE LSTM (PINN) ---
                update_progress(30, "Modèle LSTM détecté. Préparation des séquences...")
                
                scalers = model_package['scalers']
                sequence_length = model.input_shape[1] # Récupérer la longueur de séquence du modèle
                
                X_seq, _, indices = create_sequences(df_for_prediction, sequence_length)
                
                update_progress(50, "Normalisation et Prédiction VRP...")
                X_scaled = scalers['X'].transform(X_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
                y_pred_scaled = model.predict(X_scaled)
                y_pred_vrp = scalers['y'].inverse_transform(y_pred_scaled).flatten()
                
                update_progress(80, "Interprétation des prédictions VRP en labels...")
                y_true_vrp_for_derivation = df_for_prediction['VRP'].values[sequence_length-1:-1]
                y_pred_labels = derive_state_from_forecast(y_true_vrp_for_derivation, y_pred_vrp)
                
                prediction_series = pd.Series(y_pred_labels, index=indices, name='Prediction')

            else:
                # --- CHEMIN A : LOGIQUE POUR LES CLASSIFIEURS (INCHANGÉE) ---
                update_progress(30, "Modèle classifieur détecté. Extraction des features...")
                
                feature_config = config['training_config']['feature_config']
                external_features = extract_features(df_for_prediction.copy(), feature_config)
                if is_cancelled(): return None

                update_progress(75, "Alignement et Prédiction des labels...")
                model_input_features = external_features.reindex(columns=model.feature_names_in_, fill_value=0)
                predictions = model.predict(model_input_features)
                
                prediction_series = pd.Series(predictions, index=model_input_features.index, name='Prediction')

            # --- ASSEMBLAGE FINAL (COMMUN AUX DEUX CHEMINS) ---
            update_progress(95, "Assemblage des résultats finaux...")
            results_df = df_for_prediction.set_index('Date').copy()
            results_df = results_df.join(prediction_series)
            
            return results_df.reset_index()

        def _on_predicted(results_df):
            if results_df is not None:
                self.status_updated.emit("Prédiction externe terminée.")
                self.external_prediction_finished.emit(results_df)

        return self._run_in_thread(_predict, _on_predicted)

    # --- 4-6. AUTRES MÉTHODES (optimisations légères) ---
    def save_current_model(self, model_name: str):
        """Sauvegarde optimisée avec validation"""
        if not self._validate_model_save(model_name):
            return
            
        self.status_updated.emit(f"Sauvegarde du modèle '{model_name}'...")

        def _save_model(update_progress, is_cancelled):
            # On peut même ajouter un petit message de statut pour la cohérence
            update_progress(50, "Écriture des fichiers sur le disque...")
            if is_cancelled(): return False # Vérification de l'annulation (bonne pratique)

            success = save_model(model_name, self.last_training_results, self.last_feature_config)
            
            if success:
                update_progress(100, "Sauvegarde terminée.")
            
            return success
        
        def _on_model_saved(success):
            if success:
                self.error_occurred.emit("Succès", f"Le modèle '{model_name}' a été sauvegardé.")

                self.refresh_model_lists()
            else:
                self.error_occurred.emit("Erreur", "La sauvegarde du modèle a échoué.")
        _ = self._run_in_thread(_save_model, _on_model_saved)

    def _validate_model_save(self, model_name: str) -> bool:
        """Valide les conditions pour sauvegarder un modèle"""
        if not model_name:
            self.error_occurred.emit("Avertissement", "Veuillez donner un nom au modèle.")
            return False
        
        if self.last_training_results is None or self.last_feature_config is None:
            self.error_occurred.emit("Erreur", "Aucun modèle n'a été entraîné.")
            return False
        
        return True

    def start_simulation(self, params: dict) -> Optional[WorkerThread]:
        """
        Prépare la simulation en arrière-plan et retourne le worker.
        """
        if self.simulation_state != "STOPPED":
            self.error_occurred.emit("Avertissement", "Une simulation est déjà en cours.")
            return None

        def _prepare_simulation_task(update_progress: Callable, is_cancelled: Callable):
            """Tâche interne qui gère la préparation et la progression."""
            update_progress(5, "Initialisation de la simulation...")
            model_name = params['model_name']
            model, config = load_model_and_config(model_name)
            if model is None:
                raise ValueError(f"Impossible de charger le modèle '{model_name}'")
            if is_cancelled(): return None

            update_progress(20, "Génération des données synthétiques...")
            num_points = int((params['duration_days'] * 24 * 60) / 5)
            data = generate_realistic_synthetic_data(
                num_points=num_points,
                num_events_total=params['num_events'],
                ratio_long_events=params['ratio_long_events'],
                sampling_minutes=5
            )
            if is_cancelled(): return None

            update_progress(50, "Définition des cibles pour la simulation...")
            data = define_internal_event_cycle(data, pre_event_ratio=0.3, post_event_ratio=0.2)
            if is_cancelled(): return None

            update_progress(60, "Pré-calcul des caractéristiques pour la simulation...")
            def feature_progress_callback(val, msg):
                update_progress(60 + int(val * 0.35), msg)

            features = extract_features(
                data.copy(), 
                config['training_config'],
                update_progress=feature_progress_callback,
                is_cancelled=is_cancelled
            )
            if is_cancelled() or features is None: return None
            
            update_progress(100, "Préparation terminée.")
            return model, config, data, features, model_name
        
        return self._run_in_thread(_prepare_simulation_task, self._on_simulation_ready)
        
    def _on_simulation_ready(self, result):
 
        if result is None:
            self.status_updated.emit("Préparation de la simulation annulée.")
            return

        model, config, data, features, model_name = result
        
        self.simulation_model = model
        self.simulation_model_config = config

        data_indexed = data.set_index('Date')
        common_index = data_indexed.index.intersection(features.index)
        
        self.simulation_data = data_indexed.loc[common_index]
        self.simulation_feature_matrix = features.loc[common_index]
        
        self._start_simulation_timer(model_name)

    def _start_simulation_timer(self, model_name: str):
        """Démarre le timer de simulation"""
        self.simulation_index = 0
        self.simulation_state = "RUNNING"
        self.simulation_timer.start(self.simulation_speed_ms)
        
        init_payload = {
            'dates': self.simulation_data.index.tolist(), 
            'vrp': self.simulation_data['VRP'].tolist(),
            'true_labels': self.simulation_data['Ramp'].tolist(),
            'model_classes': self.simulation_model.classes_.tolist()
        }
        self.simulation_started.emit(init_payload)
        self.status_updated.emit(f"Simulation démarrée avec le modèle '{model_name}'.")

    def pause_resume_simulation(self):
        if self.simulation_state == "RUNNING":
            self.simulation_timer.stop()
            self.simulation_state = "PAUSED"
            self.status_updated.emit("Simulation en pause.")
        elif self.simulation_state == "PAUSED":
            self.simulation_timer.start(self.simulation_speed_ms)
            self.simulation_state = "RUNNING"
            self.status_updated.emit("Reprise de la simulation.")

    def stop_simulation(self):
        if self.simulation_state != "STOPPED":
            self.simulation_timer.stop()
            self.simulation_state = "STOPPED"
            self.simulation_index = 0
            self.simulation_data = None
            self.simulation_model = None
            self.simulation_model_config = None
            self.simulation_feature_matrix = None
            self.simulation_finished.emit("Simulation arrêtée par l'utilisateur.")
            self.status_updated.emit("Simulation arrêtée.")

    def _simulation_step(self):
        """Étape de simulation optimisée (logique existante conservée)"""
        if self.simulation_state != "RUNNING" or self.simulation_feature_matrix is None:
            return

        if self.simulation_index >= len(self.simulation_feature_matrix) - 1:
            self.stop_simulation()
            self.simulation_finished.emit("Fin de la simulation.")
            return

        current_features = self.simulation_feature_matrix.iloc[[self.simulation_index]]
        model_features = self.simulation_model.feature_names_in_
        current_features = current_features.reindex(columns=model_features, fill_value=0)

        prediction = self.simulation_model.predict(current_features)[0]
        probabilities = self.simulation_model.predict_proba(current_features)[0]
        
        step_payload = {
            'index': self.simulation_index,
            'prediction': prediction,
            'probabilities': probabilities.tolist()
        }
        self.simulation_step_updated.emit(step_payload)
        self.simulation_index += 1

    def explain_modified_instance(self, modified_instance: pd.DataFrame):
        """
        Prend une instance modifiée par l'utilisateur, la prédit, l'explique,
        et émet les nouveaux résultats.
        """
        if self.current_explainer is None:
            return # Pas d'explainer chargé

        model_features = self.current_explainer.model.feature_names_in_
        modified_instance = modified_instance.reindex(columns=model_features, fill_value=0)

    
        shap_values, base_value, instance_data = self.current_explainer.explain_instance(modified_instance)
        class_names = self.current_explainer.model.classes_.tolist()
        
        self.explanation_ready.emit(shap_values, base_value, instance_data, class_names)

   