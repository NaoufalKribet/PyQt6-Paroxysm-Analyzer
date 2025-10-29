# Core/feature_extractor_optimized.py - Version avec cohérence des types

import pandas as pd
import numpy as np
from scipy.stats import linregress, kurtosis, skew
from typing import List, Dict, Optional, Tuple, Callable, Union
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
from ui_dialogs import ProcessingDialog
import re

# Import de la fonction de segmentation depuis le module voisin.
from .data_processor import split_dataframe_on_gaps

def _dummy_update_progress(value: int, message: str):
    """Callback de progression qui ne fait rien."""
    pass

def _dummy_is_cancelled() -> bool:
    """Callback d'annulation qui ne s'annule jamais."""
    return False


def add_algorithmic_state_features(df: pd.DataFrame, 
                                   activity_window: int = 120, 
                                   activity_std_threshold: float = 2.0) -> pd.DataFrame:
    """
    Ajoute des caractéristiques d'état basées sur des seuils dynamiques, SANS fuite de données.
    Cette fonction est causalement correcte et peut être utilisée avant l'entraînement.

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant au moins la colonne 'VRP'.
        activity_window (int): Taille de la fenêtre (en points) pour calculer la normale locale.
        activity_std_threshold (float): Nombre d'écarts-types au-dessus de la médiane pour 
                                        considérer le signal comme "actif".

    Returns:
        pd.DataFrame: Le DataFrame original enrichi de nouvelles colonnes d'état.
    """
    print("--- Calcul des caractéristiques d'état algorithmiques (SANS fuite de données) ---")
    df_out = df.copy()
    if 'VRP' not in df_out.columns:
        raise ValueError("La colonne 'VRP' est requise.")
    
    vrp_series = df_out['VRP']

    # 1. Définir le seuil d'activité dynamique pour chaque point
    long_rolling = vrp_series.rolling(window=activity_window, min_periods=activity_window // 4)
    median = long_rolling.median().fillna(method='bfill').fillna(method='ffill')
    std = long_rolling.std().fillna(method='bfill').fillna(method='ffill').replace(0, 1e-6)
    activity_threshold = median + (activity_std_threshold * std)

    # 2. Déterminer les périodes d'activité algorithmique
    is_active = (vrp_series > activity_threshold)

    # 3. Identifier les blocs d'activité et calculer le temps écoulé
    is_active_start = (is_active & ~is_active.shift(1, fill_value=False))
    active_block_id = is_active_start.cumsum()
    active_blocks = active_block_id[is_active]
    
    time_in_state = active_blocks.groupby(active_blocks).cumcount()
    
    # 4. Calculer l'énergie cumulative dans ces blocs
    vrp_active = vrp_series[is_active]
    energy_in_state = vrp_active.groupby(active_blocks).cumsum()

    # 5. Assigner les nouvelles caractéristiques au DataFrame de sortie
    df_out['time_in_active_state'] = time_in_state
    df_out['energy_in_active_state'] = energy_in_state

    # Remplir les NaNs (périodes inactives) avec 0, car le temps et l'énergie sont nuls
    df_out['time_in_active_state'] = df_out['time_in_active_state'].fillna(0)
    df_out['energy_in_active_state'] = df_out['energy_in_active_state'].fillna(0)
    
    print(f"{is_active.sum()} points détectés comme 'actifs' par l'algorithme.")
    print("Caractéristiques 'time_in_active_state' et 'energy_in_active_state' créées.")
    print("-------------------------------------------------------------------")
    return df_out
class FeatureCalculator:
    """Calculateur de features optimisé avec cache et parallélisation"""
    
    def __init__(self, max_cache_size: int = 128):
        self.cache_size = max_cache_size
        self._rolling_cache = {}
    
    @lru_cache(maxsize=64)
    def _get_rolling_window(self, series_id: str, window: int, min_periods: int) -> pd.core.window.rolling.Rolling:
        """Cache les objets rolling pour éviter la recréation"""
        pass
    
    def calculate_slope_vectorized(self, series: pd.Series, window: Union[str, int]) -> pd.Series:
        """
        Version vectorisée du calcul de pente - ACCEPTE BOTH STR ET INT
        """
        def fast_slope(x: np.ndarray) -> float:
            if len(x) < 2 or np.isnan(x).sum() > len(x) // 2:
                return np.nan
            
            n = len(x)
            idx = np.arange(n)
            
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return np.nan
                
            x_clean = x[mask]
            idx_clean = idx[mask]
            
            n_clean = len(x_clean)
            sum_x = np.sum(idx_clean)
            sum_y = np.sum(x_clean)
            sum_xy = np.sum(idx_clean * x_clean)
            sum_x2 = np.sum(idx_clean * idx_clean)
            
            denominator = n_clean * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
                
            return (n_clean * sum_xy - sum_x * sum_y) / denominator
        
        # Calculer min_periods basé sur le type de window
        if isinstance(window, str):
            min_periods = 2  # Pour les fenêtres temporelles
        else:
            min_periods = max(2, window // 4)  # Pour les fenêtres en points
        
        return series.rolling(window=window, min_periods=min_periods).apply(fast_slope, raw=True)
    
    def calculate_multiple_rolling_stats(self, series: pd.Series, windows: List[Union[str, int]], 
                                       stats: List[str]) -> Dict[str, pd.Series]:
        """Calcule plusieurs statistiques sur plusieurs fenêtres en une seule passe"""
        results = {}
        
        # Pré-calculer tous les objets rolling nécessaires
        rolling_objects = {}
        for w in windows:
            if isinstance(w, str):
                min_periods = 1
            else:
                min_periods = max(1, w//4)
            rolling_objects[w] = series.rolling(window=w, min_periods=min_periods)
        
        # Calculer toutes les stats demandées
        stat_functions = {
            'mean': lambda r: r.mean(),
            'median': lambda r: r.median(),
            'std': lambda r: r.std(),
            'min': lambda r: r.min(),
            'max': lambda r: r.max(),
            'var': lambda r: r.var(),
            'sum': lambda r: r.sum()
        }
        
        for window in windows:
            rolling_obj = rolling_objects[window]
            window_key = str(window).replace('min', '').replace('s', '') if isinstance(window, str) else window
            for stat in stats:
                if stat in stat_functions:
                    key = f"{stat}{window_key}"
                    results[key] = stat_functions[stat](rolling_obj)
        
        return results
    
    def calculate_advanced_stats_batch(self, series: pd.Series, windows: List[Union[str, int]]) -> Dict[str, pd.Series]:
        """Calcule les statistiques avancées par batch pour plus d'efficacité"""
        results = {}
        
        for window in windows:
            # Calculer min_periods approprié
            if isinstance(window, str):
                min_periods = 3
                window_key = str(window).replace('min', '').replace('s', '')
            else:
                min_periods = max(3, window//4)
                window_key = window
            
            rolling_obj = series.rolling(window=window, min_periods=min_periods)
            
            # Kurtosis et skewness optimisés
            try:
                results[f'kurtosis{window_key}'] = rolling_obj.apply(
                    lambda x: kurtosis(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) >= 4 else np.nan,
                    raw=True
                )
                results[f'skewness{window_key}'] = rolling_obj.apply(
                    lambda x: skew(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) >= 3 else np.nan,
                    raw=True
                )
            except Exception as e:
                print(f"Warning: Advanced stats calculation failed for window {window}: {e}")
                # Fallback avec des valeurs nulles
                results[f'kurtosis{window_key}'] = pd.Series(np.nan, index=series.index)
                results[f'skewness{window_key}'] = pd.Series(np.nan, index=series.index)
        
        return results


class ExpertFeaturesCalculator:
    """Calculateur de features expertes optimisé avec types cohérents"""
    
    @staticmethod
    def calculate_thermal_momentum_vectorized(series: pd.Series, window_points: int, sampling_rate_minutes: int = 5) -> pd.Series:
        """
        CORRIGÉ : Calcul du momentum thermique robuste
        """
        try:
            # Shift en nombre de périodes (entier)
            shift_periods = int(window_points)
            shifted_series = series.shift(periods=shift_periods)
            
            # S'assurer que les deux séries sont numériques
            if shifted_series is None or len(shifted_series) == 0:
                return pd.Series(0.0, index=series.index)
                
            # Calcul du momentum avec gestion des erreurs
            numerator = series - shifted_series
            result = numerator / float(window_points)
            
            return result.fillna(0.0)
            
        except Exception as e:
            print(f"Warning in thermal_momentum calculation: {e}")
            # Fallback: retourner une série de zéros
            return pd.Series(0.0, index=series.index)
    
    @staticmethod
    def calculate_energy_buildup_ratio_vectorized(series: pd.Series, short_window: Union[str, int], 
                                                long_window: Union[str, int]) -> pd.Series:
        """
        CORRIGÉ : Accepte les deux types mais assure la cohérence
        """
        short_mean = series.rolling(short_window, min_periods=1).mean()
        long_mean = series.rolling(long_window, min_periods=1).mean()
        
        return short_mean / np.maximum(long_mean, 1e-6)
    
    @staticmethod
    def calculate_acceleration_features(features_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcule toutes les accélérations en une fois"""
        acceleration_features = {}
        
        acceleration_columns = [col for col in features_df.columns 
                              if any(base in col for base in ['slope', 'range', 'std'])]
        
        for col in acceleration_columns:
            if col in features_df.columns:
                acceleration_features[f'{col}_accel'] = features_df[col].diff()
        
        return acceleration_features

    @staticmethod
    def calculate_zscore_vectorized(series: pd.Series, window: Union[str, int]) -> pd.Series:
        """
        Calcule le Z-Score glissant pour normaliser le signal par rapport à son histoire locale.
        Rend la feature indépendante de l'échelle absolue de l'événement.
        """
        # min_periods est crucial pour ne pas avoir de zéros au début
        min_periods = int(window) // 4 if isinstance(window, int) else 20 
        
        rolling_stats = series.rolling(window, min_periods=min_periods)
        mean = rolling_stats.mean()
        std = rolling_stats.std()
        
        # Remplacer les std nuls ou très faibles pour éviter la division par zéro
        safe_std = std.replace(0, np.nan).fillna(method='ffill').fillna(1e-6)
        
        z_score = (series - mean) / safe_std
        return z_score.fillna(0.0) # Les NaNs au début sont remplacés par 0

    @staticmethod
    def calculate_range_position_vectorized(series: pd.Series, window: Union[str, int]) -> pd.Series:
        """
        Calcule la position relative du signal (entre 0 et 1) dans son étendue locale.
        Une valeur de 1 signifie que le signal est à son maximum sur la fenêtre.
        """
        min_periods = int(window) // 4 if isinstance(window, int) else 20

        rolling_window = series.rolling(window, min_periods=min_periods)
        min_val = rolling_window.min()
        max_val = rolling_window.max()
        
        range_val = max_val - min_val
        # Éviter la division par zéro si la fenêtre est plate
        safe_range = range_val.replace(0, 1e-6)
        
        position = (series - min_val) / safe_range
        return position.fillna(0.0) # Remplir les NaNs initiaux

    @staticmethod
    def calculate_interquantile_range_vectorized(series: pd.Series, window: Union[str, int], 
                                               low_q: float = 0.1, high_q: float = 0.9) -> pd.Series:
        """
        Calcule l'étendue interquantile, une mesure de volatilité robuste au bruit et aux outliers.
        """
        min_periods = int(window) // 4 if isinstance(window, int) else 20

        rolling_window = series.rolling(window, min_periods=min_periods)
        
        q_low = rolling_window.quantile(low_q)
        q_high = rolling_window.quantile(high_q)
        
        iqr = q_high - q_low
        return iqr.fillna(0.0)
    
    @staticmethod
    def calculate_volatility_ratios_batch(features_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcule tous les ratios de volatilité de manière optimisée"""
        ratios = {}
        
        std_columns = [col for col in features_df.columns if 'std' in col]
        windows = []
        
        # Extraction plus robuste des numéros de fenêtres
        for col in std_columns:
            numbers = re.findall(r'\d+', col)
            if numbers:
                windows.extend([int(num) for num in numbers])
        
        windows = sorted(list(set(windows)))  # Unique et trié
        
        # Calculer tous les ratios possibles entre fenêtres courtes et longues
        for i, short_win in enumerate(windows[:-1]):
            for long_win in windows[i+1:]:
                short_col = f'std{short_win}'
                long_col = f'std{long_win}'
                
                if short_col in features_df.columns and long_col in features_df.columns:
                    ratio_name = f'volatility_ratio_{short_win}_{long_win}'
                    ratios[ratio_name] = features_df[short_col] / np.maximum(features_df[long_col], 1e-6)
        
        return ratios


def calculate_eruption_precursor_score_optimized(features_df: pd.DataFrame) -> pd.Series:
    """Version optimisée du score composite utilisant numpy vectorisé"""
    # Initialiser le score avec des zéros
    score = np.zeros(len(features_df))
    
    # Définir les poids et features de manière optimisée
    feature_weights = {
        'slope30_accel': 0.3,
        'volatility_ratio_10_50': 0.3,
        'range10_accel': 0.2,
        'kurtosis30': 0.1
    }
    
    # Calcul vectorisé de tous les composants du score
    for feature, weight in feature_weights.items():
        if feature in features_df.columns:
            values = features_df[feature].fillna(0).values
            
            if feature == 'volatility_ratio_10_50':
                # Normalisation autour de 0
                values = values - 1
            elif feature == 'kurtosis30':
                # Clipping pour éviter la domination
                values = np.clip(values, -5, 5)
            
            score += weight * values
    
    return pd.Series(score, index=features_df.index, dtype=np.float32)


class WindowManager:
    """Gestionnaire centralisé des fenêtres pour éviter les incohérences"""
    
    def __init__(self, config: Dict, sampling_rate_minutes: int = 5):
        self.config = config
        self.sampling_rate = sampling_rate_minutes
        self.windows_points = self._extract_windows_from_config()
        self.windows_mapping = self._create_windows_mapping()
    
    def _extract_windows_from_config(self) -> List[int]:
        """Extrait les tailles de fenêtres uniques de la configuration"""
        windows = set()
        
        # 1. Extraction depuis les SpinBox de l'UI
        for key, value in self.config.get('window_sizes', {}).items():
            if isinstance(value, int):
                windows.add(value)
        
        # 2. Extraction depuis les noms de features
        for feature in self.config.get('features', []):
            found_numbers = re.findall(r'\d+', feature)
            for num_str in found_numbers:
                windows.add(int(num_str))
        
        return sorted(list(windows))
    
    def _create_windows_mapping(self) -> Dict[int, Dict[str, Union[str, int]]]:
        """Crée un mapping cohérent entre points et temps"""
        mapping = {}
        for points in self.windows_points:
            time_str = f"{points * self.sampling_rate}min"
            mapping[points] = {
                'points': points,
                'time_str': time_str,
                'min_periods_basic': max(1, points // 4),
                'min_periods_advanced': max(3, points // 4)
            }
        return mapping
    
    def get_window_info(self, points: int) -> Dict:
        """Retourne les infos complètes pour une fenêtre donnée"""
        return self.windows_mapping.get(points, {
            'points': points,
            'time_str': f"{points * self.sampling_rate}min",
            'min_periods_basic': max(1, points // 4),
            'min_periods_advanced': max(3, points // 4)
        })


def extract_features_optimized(
    df: pd.DataFrame, 
    config: Dict, 
    max_gap_str: str = "1 day",
    update_progress: Callable = _dummy_update_progress,
    is_cancelled: Callable = _dummy_is_cancelled
) -> Optional[pd.DataFrame]:
    
    print("🚀 Démarrage de l'extraction de features (version avec état algorithmique).")
    
    # 1. AJOUT DES FEATURES D'ÉTAT SANS FUITE DE DONNÉES
    # Cette étape est cruciale et se fait en amont sur le DataFrame complet.
    try:
        df_with_state = add_algorithmic_state_features(df)
    except Exception as e:
        print(f"❌ ERREUR critique lors du calcul des features d'état : {e}")
        return None

    # 2. INITIALISATION DES CALCULATEURS (inchangé)
    safe_config = config.copy()
    feature_calc = FeatureCalculator()
    expert_calc = ExpertFeaturesCalculator()
    window_manager = WindowManager(safe_config)
    
    # 3. SEGMENTATION (se fait maintenant sur le DataFrame enrichi)
    try:
        max_gap_duration = pd.to_timedelta(max_gap_str)
        # La fonction de segmentation préserve l'index et les nouvelles colonnes
        segments = split_dataframe_on_gaps(df_with_state, max_gap_duration)
    except Exception as e:
        print(f"⚠️ Erreur de segmentation : {e}. Traitement en un seul segment.")
        segments = [df_with_state]

    # 4. TRAITEMENT PARALLÈLE DES SEGMENTS (inchangé)
    if len(segments) > 1:
        print(f"📊 Traitement parallèle de {len(segments)} segments...")
        # Simulé pour la simplicité, vous pouvez réactiver votre ThreadPoolExecutor ici
        processed_segments = [_process_single_segment(s, safe_config, feature_calc, expert_calc, window_manager) for s in segments]
    else:
        print("📊 Traitement d'un seul segment...")
        processed_segments = [_process_single_segment(segments[0], safe_config, feature_calc, expert_calc, window_manager)]
    
    valid_segments = [seg for seg in processed_segments if seg is not None and not seg.empty]
    if not valid_segments:
        print("❌ Aucun segment valide traité, l'extraction a échoué.")
        return None
    
    # 5. CONCATÉNATION ET FINALISATION
    print("🔗 Assemblage final des segments...")
    final_features_df = pd.concat(valid_segments, ignore_index=False, sort=False)
    
    # Sélection finale des caractéristiques demandées par l'utilisateur
    final_features_df = _finalize_feature_selection(final_features_df, safe_config)
    
    print(f"✅ Extraction terminée : {len(final_features_df)} lignes × {final_features_df.shape[1]} features")
    
    return final_features_df



def _process_segments_parallel(segments: List[pd.DataFrame], config: Dict, 
                             feature_calc: FeatureCalculator, 
                             expert_calc: ExpertFeaturesCalculator,
                             window_manager: WindowManager) -> List[pd.DataFrame]:
    """Traite les segments en parallèle avec configuration sécurisée"""
    
    processed_segments = []
    
    # Utiliser ThreadPoolExecutor pour la parallélisation
    with ThreadPoolExecutor(max_workers=min(4, len(segments))) as executor:
        # Soumettre tous les segments avec des copies de config
        future_to_segment = {}
        for i, segment in enumerate(segments):
            # Chaque thread reçoit sa propre copie des objets
            thread_config = config.copy()
            thread_feature_calc = FeatureCalculator()
            thread_expert_calc = ExpertFeaturesCalculator()
            thread_window_manager = WindowManager(thread_config)
            
            future = executor.submit(
                _process_single_segment, 
                segment, 
                thread_config, 
                thread_feature_calc, 
                thread_expert_calc, 
                thread_window_manager
            )
            future_to_segment[future] = i
        
        # Récupérer les résultats dans l'ordre
        results = [None] * len(segments)
        for future in as_completed(future_to_segment):
            segment_idx = future_to_segment[future]
            try:
                result = future.result()
                results[segment_idx] = result
                print(f"  ✓ Segment {segment_idx + 1} traité")
            except Exception as e:
                print(f"  ❌ Erreur segment {segment_idx + 1}: {e}")
                results[segment_idx] = None
        
        processed_segments = [r for r in results if r is not None]
    
    return processed_segments


# Dans Core/feature_extractor.py (remplacez la fonction existante par celle-ci)

# Dans Core/feature_extractor.py, REMPLACEZ la fonction _process_single_segment existante par celle-ci

def _process_single_segment(segment_df: pd.DataFrame, config: Dict,
                          feature_calc: FeatureCalculator,
                          expert_calc: ExpertFeaturesCalculator,
                          window_manager: WindowManager) -> Optional[pd.DataFrame]:
    """
    Traite un segment unique en calculant toutes les caractéristiques requises sur le VRP
    et en CONSERVANT les caractéristiques pré-calculées.
    VERSION FINALE ROBUSTE, avec une gestion correcte des petites fenêtres.
    """
    if 'VRP' not in segment_df.columns or segment_df.empty:
        print("    -> AVERTISSEMENT: Segment vide ou sans colonne 'VRP'. Ignoré.")
        return None

    # On part d'une copie du segment pour conserver les colonnes pré-existantes.
    features_df = segment_df.copy()
    data_series = features_df['VRP']
    
    print(f"  -> Traitement d'un segment de {len(data_series)} points. Fenêtres : {window_manager.windows_points}")

    log_data_series = np.log1p(data_series.clip(lower=0))
    print("  -> Transformation logarithmique du signal VRP effectuée.")
    # Boucle unique pour les calculs par fenêtre pour plus d'efficacité et de contrôle.
    for points in window_manager.windows_points:
        time_window_str = f"{points * 5}min"  # Assumant 5 minutes/point

        # --- CORRECTION : Logique de min_periods centralisée et plus sûre ---
        # Pour les stats de base, on peut être souple.
        min_p_base = max(1, points // 4)
        # Pour les stats avancées (kurtosis, quantiles), il faut plus de points.
        min_p_advanced = max(4, points // 2)

        # Vérifier si la fenêtre est assez grande pour les calculs avancés
        if points < 4:
            print(f"    -> Fenêtre de {points} pts trop petite pour les stats avancées. Ignorées.")
            min_p_advanced = points # S'assurer de ne pas demander plus de points que la taille de la fenêtre

        # --- ÉTAPE 1 : Calcul des composants de base ---
        rolling_base = data_series.rolling(time_window_str, min_periods=min_p_base)
        rolling_adv = data_series.rolling(time_window_str, min_periods=min_p_advanced)

        min_val = rolling_base.min()
        max_val = rolling_base.max()
        mean_val = rolling_base.mean()
        std_val = rolling_base.std()
        
        # --- ÉTAPE 2 : Assemblage des caractéristiques finales à partir des composants ---
        features_df[f'median{points}'] = rolling_base.median()
        features_df[f'std{points}'] = std_val
        features_df[f'min{points}'] = min_val
        features_df[f'max{points}'] = max_val
        
        # Caractéristiques avancées avec garde-fou
        if points >= 4:
            features_df[f'kurtosis{points}'] = rolling_adv.kurt()
            features_df[f'skewness{points}'] = rolling_adv.skew()
            q_low = rolling_adv.quantile(0.1)
            q_high = rolling_adv.quantile(0.9)
            features_df[f'iqr{points}'] = q_high - q_low
        else: # Remplir avec 0 si la fenêtre est trop petite
            features_df[f'kurtosis{points}'] = 0.0
            features_df[f'skewness{points}'] = 0.0
            features_df[f'iqr{points}'] = 0.0

        # Pente (Vitesse) et Accélération
        features_df[f'slope{points}'] = feature_calc.calculate_slope_vectorized(data_series, window=time_window_str)
        features_df[f'slope{points}_accel'] = features_df[f'slope{points}'].diff()

        # Caractéristiques de robustesse
        safe_std = std_val.replace(0, 1e-6)
        features_df[f'zscore_{points}'] = (data_series - mean_val) / safe_std

        range_val = max_val - min_val
        safe_range = range_val.replace(0, 1e-6)
        features_df[f'range_pos{points}'] = (data_series - min_val) / safe_range
        features_df[f'range{points}'] = range_val
        log_slope = feature_calc.calculate_slope_vectorized(log_data_series, window=time_window_str)
        features_df[f'log_slope{points}'] = log_slope
        
        # L'accélération de la pente du log. Si la croissance est purement exponentielle,
        # cette valeur devrait tendre vers zéro.
        features_df[f'log_slope{points}_accel'] = log_slope.diff()
    # --- ÉTAPE 3 : Calculs post-boucle ---
    print("  -> Calcul des caractéristiques composites (ratios, score)...")
    try:
        volatility_ratios = expert_calc.calculate_volatility_ratios_batch(features_df)
        features_df = features_df.assign(**volatility_ratios)
    except Exception as e:
        print(f"    -> AVERTISSEMENT: Le calcul des ratios de volatilité a échoué: {e}")
        
    features_df['precursor_score'] = calculate_eruption_precursor_score_optimized(features_df)
    
    # --- ÉTAPE 4 : NETTOYAGE FINAL ---
    features_df.fillna(0.0, inplace=True)
    
    print(f"  -> Fin du traitement du segment. {features_df.shape[1]} caractéristiques au total.")
    return features_df
def _finalize_feature_selection(features_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Finalise la sélection en appliquant la liste de l'utilisateur comme un filtre final.
    L'extraction a déjà calculé un sur-ensemble de caractéristiques pour gérer les dépendances.
    """
    
    # 1. Obtenir la liste EXACTE des features que l'utilisateur veut dans son jeu de données final.
    user_final_selection = config.get('features', [])
    
    # 2. Sécurité : si la liste est vide, on retourne un DataFrame vide pour éviter les erreurs.
    if not user_final_selection:
        print("AVERTISSEMENT: Aucune caractéristique n'a été sélectionnée. Retour d'un DataFrame vide.")
        return pd.DataFrame()

    # 3. Utiliser .reindex() pour créer le DataFrame final. C'est la méthode la plus robuste.
    #    - Elle sélectionne DANS features_df UNIQUEMENT les colonnes présentes dans votre liste.
    #    - Si une colonne que vous avez demandée n'a pas pu être calculée (ex: dépendance non satisfaite),
    #      elle sera quand même créée et remplie avec 0.0, évitant ainsi un crash.
    final_df = features_df.reindex(columns=user_final_selection, fill_value=0.0)
    
    # 4. Afficher un message clair et retourner le DataFrame final.
    print(f"INFO: Sélection finale de {final_df.shape[1]} caractéristiques, strictement basée sur la sélection de l'utilisateur.")
    
    return final_df.astype(np.float32)

# Fonction de compatibilité avec l'ancienne API
def extract_features(df: pd.DataFrame, config: Dict, **kwargs) -> Optional[pd.DataFrame]:
    """Interface de compatibilité qui passe les kwargs à la fonction optimisée."""
    return extract_features_optimized(df, config, **kwargs)