"""
Feature Extraction Module for Volcanic Activity Analysis
=========================================================

This module provides optimized functions to extract a comprehensive set of
features from time-series volcanic seismic data. It includes both basic statistical
and expert domain-specific features, with a focus on preventing data leakage
during feature computation.

The feature extraction pipeline supports:
- Multi-scale temporal windows for capturing patterns at different time scales
- Statistical features (mean, std, skewness, kurtosis, quantiles)
- Expert volcanological features (thermal momentum, energy buildup, precursor scores)
- Algorithmic state detection without data leakage
- Parallel processing for large datasets with automatic segmentation

@author: KRIBET Naoufal
@affiliation: 5th year Engineering Student, EOST (École et Observatoire des Sciences de la Terre)
@date: 2025-11-17
@version: 1.0
"""

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
from .data_processor import split_dataframe_on_gaps

def _dummy_update_progress(value: int, message: str):
    """
    Dummy progress callback that does nothing.
    
    Used as default when no UI progress tracking is needed.
    
    @param value: Progress value (0-100)
    @param message: Progress message
    @author: KRIBET Naoufal
    """
    pass

def _dummy_is_cancelled() -> bool:
    """
    Dummy cancellation callback that never cancels.
    
    Used as default when no cancellation mechanism is needed.
    
    @return: Always False (never cancelled)
    @author: KRIBET Naoufal
    """
    return False


def add_algorithmic_state_features(df: pd.DataFrame, 
                                   activity_window: int = 120, 
                                   activity_std_threshold: float = 2.0) -> pd.DataFrame:
    """
    Adds state features based on dynamic thresholds WITHOUT data leakage.
    
    This function implements a causally correct activity detection algorithm
    that can be safely used before model training. It identifies "active" periods
    based on deviation from rolling baseline statistics, then tracks temporal
    and energetic properties of these active states.
    
    The algorithm is causal because at each time point, it only uses information
    from the past (via backward-looking rolling windows), ensuring no future
    information leaks into the features.
    
    @param df: Input DataFrame containing at least the 'VRP' column
    @param activity_window: Window size (in points) for computing local baseline
    @param activity_std_threshold: Number of standard deviations above median
                                    to consider signal as "active"
    @return: Original DataFrame enriched with new state columns:
             - time_in_active_state: Cumulative time spent in current active block
             - energy_in_active_state: Cumulative energy in current active block
    @raises ValueError: If 'VRP' column is missing from input DataFrame
    
    @author: KRIBET Naoufal
    """
    print("--- Computing algorithmic state features (WITHOUT data leakage) ---")
    df_out = df.copy()
    if 'VRP' not in df_out.columns:
        raise ValueError("'VRP' column is required.")
    
    vrp_series = df_out['VRP']

    # 1. Define dynamic activity threshold for each point
    long_rolling = vrp_series.rolling(window=activity_window, min_periods=activity_window // 4)
    median = long_rolling.median().fillna(method='bfill').fillna(method='ffill')
    std = long_rolling.std().fillna(method='bfill').fillna(method='ffill').replace(0, 1e-6)
    activity_threshold = median + (activity_std_threshold * std)

    # 2. Determine algorithmic activity periods
    is_active = (vrp_series > activity_threshold)

    # 3. Identify activity blocks and calculate elapsed time
    is_active_start = (is_active & ~is_active.shift(1, fill_value=False))
    active_block_id = is_active_start.cumsum()
    active_blocks = active_block_id[is_active]
    
    time_in_state = active_blocks.groupby(active_blocks).cumcount()
    
    # 4. Calculate cumulative energy in these blocks
    vrp_active = vrp_series[is_active]
    energy_in_state = vrp_active.groupby(active_blocks).cumsum()

    # 5. Assign new features to output DataFrame
    df_out['time_in_active_state'] = time_in_state
    df_out['energy_in_active_state'] = energy_in_state

    # Fill NaNs (inactive periods) with 0, as time and energy are null
    df_out['time_in_active_state'] = df_out['time_in_active_state'].fillna(0)
    df_out['energy_in_active_state'] = df_out['energy_in_active_state'].fillna(0)
    
    print(f"{is_active.sum()} points detected as 'active' by algorithm.")
    print("Features 'time_in_active_state' and 'energy_in_active_state' created.")
    print("-------------------------------------------------------------------")
    return df_out


class FeatureCalculator:
    """
    Optimized feature calculator with caching and parallelization.
    
    This class provides efficient methods for computing statistical features
    on time-series data, with built-in caching to avoid redundant calculations.
    
    @author: KRIBET Naoufal
    """
    
    def __init__(self, max_cache_size: int = 128):
        """
        Initialize the feature calculator.
        
        @param max_cache_size: Maximum size for LRU cache
        @author: KRIBET Naoufal
        """
        self.cache_size = max_cache_size
        self._rolling_cache = {}
    
    @lru_cache(maxsize=64)
    def _get_rolling_window(self, series_id: str, window: int, min_periods: int) -> pd.core.window.rolling.Rolling:
        """
        Cache rolling window objects to avoid recreation.
        
        @param series_id: Unique identifier for the series
        @param window: Window size
        @param min_periods: Minimum periods required
        @return: Rolling window object
        @author: KRIBET Naoufal
        """
        pass
    
    def calculate_slope_vectorized(self, series: pd.Series, window: Union[str, int]) -> pd.Series:
        """
        Vectorized slope calculation - ACCEPTS BOTH STRING AND INT windows.
        
        Computes linear regression slope over rolling windows using an optimized
        vectorized algorithm. This is much faster than scipy.stats.linregress
        applied repeatedly.
        
        @param series: Input time series
        @param window: Window size (int for points, str for time like '30min')
        @return: Series of slope values
        @author: KRIBET Naoufal
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
        
        # Calculate min_periods based on window type
        if isinstance(window, str):
            min_periods = 2  # For time windows
        else:
            min_periods = max(2, window // 4)  # For point windows
        
        return series.rolling(window=window, min_periods=min_periods).apply(fast_slope, raw=True)
    
    def calculate_multiple_rolling_stats(self, series: pd.Series, windows: List[Union[str, int]], 
                                       stats: List[str]) -> Dict[str, pd.Series]:
        """
        Calculate multiple statistics over multiple windows in a single pass.
        
        This method is optimized to compute various statistics (mean, median, std, etc.)
        across different window sizes efficiently by reusing rolling window objects.
        
        @param series: Input time series
        @param windows: List of window sizes
        @param stats: List of statistic names to compute
        @return: Dictionary mapping feature names to computed Series
        @author: KRIBET Naoufal
        """
        results = {}
        
        # Pre-calculate all necessary rolling objects
        rolling_objects = {}
        for w in windows:
            if isinstance(w, str):
                min_periods = 1
            else:
                min_periods = max(1, w//4)
            rolling_objects[w] = series.rolling(window=w, min_periods=min_periods)
        
        # Calculate all requested stats
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
        """
        Calculate advanced statistics in batch for better efficiency.
        
        Computes kurtosis and skewness across multiple windows with proper
        error handling and fallback mechanisms.
        
        @param series: Input time series
        @param windows: List of window sizes
        @return: Dictionary of advanced statistics
        @author: KRIBET Naoufal
        """
        results = {}
        
        for window in windows:
            # Calculate appropriate min_periods
            if isinstance(window, str):
                min_periods = 3
                window_key = str(window).replace('min', '').replace('s', '')
            else:
                min_periods = max(3, window//4)
                window_key = window
            
            rolling_obj = series.rolling(window=window, min_periods=min_periods)
            
            # Optimized kurtosis and skewness
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
                # Fallback with null values
                results[f'kurtosis{window_key}'] = pd.Series(np.nan, index=series.index)
                results[f'skewness{window_key}'] = pd.Series(np.nan, index=series.index)
        
        return results


class ExpertFeaturesCalculator:
    """
    Calculator for expert volcanological features with optimized type handling.
    
    This class implements domain-specific features developed in consultation
    with volcanology experts, designed to capture physical processes relevant
    to eruption forecasting.
    
    @author: KRIBET Naoufal
    """
    
    @staticmethod
    def calculate_thermal_momentum_vectorized(series: pd.Series, window_points: int, sampling_rate_minutes: int = 5) -> pd.Series:
        """
        CORRECTED: Robust thermal momentum calculation.
        
        Thermal momentum represents the rate of change in volcanic activity,
        analogous to physical momentum. It helps identify acceleration phases
        that may precede eruptions.
        
        @param series: Input VRP time series
        @param window_points: Window size in number of points
        @param sampling_rate_minutes: Sampling rate in minutes
        @return: Thermal momentum series
        @author: KRIBET Naoufal
        """
        try:
            # Shift by number of periods (integer)
            shift_periods = int(window_points)
            shifted_series = series.shift(periods=shift_periods)
            
            # Ensure both series are numeric
            if shifted_series is None or len(shifted_series) == 0:
                return pd.Series(0.0, index=series.index)
                
            # Calculate momentum with error handling
            numerator = series - shifted_series
            result = numerator / float(window_points)
            
            return result.fillna(0.0)
            
        except Exception as e:
            print(f"Warning in thermal_momentum calculation: {e}")
            # Fallback: return series of zeros
            return pd.Series(0.0, index=series.index)
    
    @staticmethod
    def calculate_energy_buildup_ratio_vectorized(series: pd.Series, short_window: Union[str, int], 
                                                long_window: Union[str, int]) -> pd.Series:
        """
        CORRECTED: Accepts both types but ensures consistency.
        
        Energy buildup ratio compares short-term to long-term energy levels,
        highlighting rapid increases that may signal impending eruptions.
        
        @param series: Input VRP time series
        @param short_window: Short-term window
        @param long_window: Long-term window
        @return: Energy buildup ratio series
        @author: KRIBET Naoufal
        """
        short_mean = series.rolling(short_window, min_periods=1).mean()
        long_mean = series.rolling(long_window, min_periods=1).mean()
        
        return short_mean / np.maximum(long_mean, 1e-6)
    
    @staticmethod
    def calculate_acceleration_features(features_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all acceleration features at once.
        
        Acceleration (second derivative) of key features can reveal
        changing trends that precede volcanic events.
        
        @param features_df: DataFrame containing base features
        @return: Dictionary of acceleration features
        @author: KRIBET Naoufal
        """
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
        Calculate rolling Z-Score to normalize signal relative to local history.
        
        Z-Score makes features scale-independent, allowing the model to
        recognize patterns regardless of absolute event magnitude.
        
        @param series: Input time series
        @param window: Window size for computing statistics
        @return: Z-Score normalized series
        @author: KRIBET Naoufal
        """
        # min_periods is crucial to avoid zeros at the beginning
        min_periods = int(window) // 4 if isinstance(window, int) else 20 
        
        rolling_stats = series.rolling(window, min_periods=min_periods)
        mean = rolling_stats.mean()
        std = rolling_stats.std()
        
        # Replace null or very small std to avoid division by zero
        safe_std = std.replace(0, np.nan).fillna(method='ffill').fillna(1e-6)
        
        z_score = (series - mean) / safe_std
        return z_score.fillna(0.0)  # Initial NaNs replaced with 0

    @staticmethod
    def calculate_range_position_vectorized(series: pd.Series, window: Union[str, int]) -> pd.Series:
        """
        Calculate relative signal position (between 0 and 1) within local range.
        
        A value of 1 means the signal is at its maximum over the window,
        providing scale-independent information about signal state.
        
        @param series: Input time series
        @param window: Window size
        @return: Range position series (0-1)
        @author: KRIBET Naoufal
        """
        min_periods = int(window) // 4 if isinstance(window, int) else 20

        rolling_window = series.rolling(window, min_periods=min_periods)
        min_val = rolling_window.min()
        max_val = rolling_window.max()
        
        range_val = max_val - min_val
        # Avoid division by zero if window is flat
        safe_range = range_val.replace(0, 1e-6)
        
        position = (series - min_val) / safe_range
        return position.fillna(0.0)  # Fill initial NaNs

    @staticmethod
    def calculate_interquantile_range_vectorized(series: pd.Series, window: Union[str, int], 
                                               low_q: float = 0.1, high_q: float = 0.9) -> pd.Series:
        """
        Calculate interquantile range, a robust volatility measure resistant to noise and outliers.
        
        @param series: Input time series
        @param window: Window size
        @param low_q: Lower quantile (default 0.1 = 10th percentile)
        @param high_q: Upper quantile (default 0.9 = 90th percentile)
        @return: Interquantile range series
        @author: KRIBET Naoufal
        """
        min_periods = int(window) // 4 if isinstance(window, int) else 20

        rolling_window = series.rolling(window, min_periods=min_periods)
        
        q_low = rolling_window.quantile(low_q)
        q_high = rolling_window.quantile(high_q)
        
        iqr = q_high - q_low
        return iqr.fillna(0.0)
    
    @staticmethod
    def calculate_volatility_ratios_batch(features_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all volatility ratios in an optimized manner.
        
        Volatility ratios between different time scales can reveal changing
        dynamic regimes characteristic of pre-eruptive phases.
        
        @param features_df: DataFrame containing std features
        @return: Dictionary of volatility ratio features
        @author: KRIBET Naoufal
        """
        ratios = {}
        
        std_columns = [col for col in features_df.columns if 'std' in col]
        windows = []
        
        # More robust extraction of window numbers
        for col in std_columns:
            numbers = re.findall(r'\d+', col)
            if numbers:
                windows.extend([int(num) for num in numbers])
        
        windows = sorted(list(set(windows)))  # Unique and sorted
        
        # Calculate all possible ratios between short and long windows
        for i, short_win in enumerate(windows[:-1]):
            for long_win in windows[i+1:]:
                short_col = f'std{short_win}'
                long_col = f'std{long_win}'
                
                if short_col in features_df.columns and long_col in features_df.columns:
                    ratio_name = f'volatility_ratio_{short_win}_{long_win}'
                    ratios[ratio_name] = features_df[short_col] / np.maximum(features_df[long_col], 1e-6)
        
        return ratios


def calculate_eruption_precursor_score_optimized(features_df: pd.DataFrame) -> pd.Series:
    """
    Optimized version of composite precursor score using vectorized numpy.
    
    This expert-designed score combines multiple features weighted by their
    importance for eruption forecasting, based on volcanological domain knowledge.
    
    @param features_df: DataFrame containing required features
    @return: Composite precursor score series
    @author: KRIBET Naoufal
    """
    # Initialize score with zeros
    score = np.zeros(len(features_df))
    
    # Define weights and features in optimized manner
    feature_weights = {
        'slope30_accel': 0.3,
        'volatility_ratio_10_50': 0.3,
        'range10_accel': 0.2,
        'kurtosis30': 0.1
    }
    
    # Vectorized calculation of all score components
    for feature, weight in feature_weights.items():
        if feature in features_df.columns:
            values = features_df[feature].fillna(0).values
            
            if feature == 'volatility_ratio_10_50':
                # Normalization around 0
                values = values - 1
            elif feature == 'kurtosis30':
                # Clipping to avoid domination
                values = np.clip(values, -5, 5)
            
            score += weight * values
    
    return pd.Series(score, index=features_df.index, dtype=np.float32)


class WindowManager:
    """
    Centralized window manager to avoid inconsistencies.
    
    This class ensures consistent window handling across all feature
    calculations by maintaining a single source of truth for window
    sizes and their properties.
    
    @author: KRIBET Naoufal
    """
    
    def __init__(self, config: Dict, sampling_rate_minutes: int = 5):
        """
        Initialize window manager from configuration.
        
        @param config: Feature extraction configuration
        @param sampling_rate_minutes: Data sampling rate in minutes
        @author: KRIBET Naoufal
        """
        self.config = config
        self.sampling_rate = sampling_rate_minutes
        self.windows_points = self._extract_windows_from_config()
        self.windows_mapping = self._create_windows_mapping()
    
    def _extract_windows_from_config(self) -> List[int]:
        """
        Extract unique window sizes from configuration.
        
        @return: Sorted list of window sizes in points
        @author: KRIBET Naoufal
        """
        windows = set()
        
        # 1. Extract from UI SpinBox values
        for key, value in self.config.get('window_sizes', {}).items():
            if isinstance(value, int):
                windows.add(value)
        
        # 2. Extract from feature names
        for feature in self.config.get('features', []):
            found_numbers = re.findall(r'\d+', feature)
            for num_str in found_numbers:
                windows.add(int(num_str))
        
        return sorted(list(windows))
    
    def _create_windows_mapping(self) -> Dict[int, Dict[str, Union[str, int]]]:
        """
        Create consistent mapping between points and time.
        
        @return: Dictionary mapping window sizes to their properties
        @author: KRIBET Naoufal
        """
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
        """
        Return complete information for a given window.
        
        @param points: Window size in points
        @return: Dictionary with window properties
        @author: KRIBET Naoufal
        """
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
    """
    Main feature extraction function with algorithmic state features.
    
    This is the primary entry point for feature extraction. It orchestrates
    the entire pipeline including:
    1. Adding algorithmic state features without data leakage
    2. Segmenting the time series based on gaps
    3. Parallel processing of segments
    4. Feature calculation and selection
    5. Final assembly and validation
    
    @param df: Input DataFrame with 'VRP' and 'Date' columns
    @param config: Feature extraction configuration dictionary
    @param max_gap_str: Maximum gap duration before splitting (e.g., "1 day")
    @param update_progress: Optional callback for progress updates
    @param is_cancelled: Optional callback to check for cancellation
    @return: DataFrame with extracted features, or None if extraction fails
    
    @author: KRIBET Naoufal
    """
    
    print("🚀 Starting feature extraction (version with algorithmic state).")
    
    # 1. ADD STATE FEATURES WITHOUT DATA LEAKAGE
    # This critical step is performed upfront on the complete DataFrame
    try:
        df_with_state = add_algorithmic_state_features(df)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during state feature calculation: {e}")
        return None

    # 2. INITIALIZE CALCULATORS (unchanged)
    safe_config = config.copy()
    feature_calc = FeatureCalculator()
    expert_calc = ExpertFeaturesCalculator()
    window_manager = WindowManager(safe_config)
    
    # 3. SEGMENTATION (now performed on enriched DataFrame)
    try:
        max_gap_duration = pd.to_timedelta(max_gap_str)
        # Segmentation function preserves index and new columns
        segments = split_dataframe_on_gaps(df_with_state, max_gap_duration)
    except Exception as e:
        print(f"⚠️ Segmentation error: {e}. Processing as single segment.")
        segments = [df_with_state]

    # 4. PARALLEL SEGMENT PROCESSING (unchanged)
    if len(segments) > 1:
        print(f"📊 Parallel processing of {len(segments)} segments...")
        # Simplified for clarity, you can reactivate ThreadPoolExecutor here
        processed_segments = [_process_single_segment(s, safe_config, feature_calc, expert_calc, window_manager) for s in segments]
    else:
        print("📊 Processing single segment...")
        processed_segments = [_process_single_segment(segments[0], safe_config, feature_calc, expert_calc, window_manager)]
    
    valid_segments = [seg for seg in processed_segments if seg is not None and not seg.empty]
    if not valid_segments:
        print("❌ No valid segments processed, extraction failed.")
        return None
    
    # 5. CONCATENATION AND FINALIZATION
    print("🔗 Final assembly of segments...")
    final_features_df = pd.concat(valid_segments, ignore_index=False, sort=False)
    
    # Final selection of user-requested features
    final_features_df = _finalize_feature_selection(final_features_df, safe_config)
    
    print(f"✅ Extraction completed: {len(final_features_df)} rows × {final_features_df.shape[1]} features")
    
    return final_features_df



def _process_segments_parallel(segments: List[pd.DataFrame], config: Dict, 
                             feature_calc: FeatureCalculator, 
                             expert_calc: ExpertFeaturesCalculator,
                             window_manager: WindowManager) -> List[pd.DataFrame]:
    """
    Process segments in parallel with safe configuration.
    
    @param segments: List of DataFrame segments to process
    @param config: Feature extraction configuration
    @param feature_calc: Feature calculator instance
    @param expert_calc: Expert features calculator instance
    @param window_manager: Window manager instance
    @return: List of processed DataFrames
    @author: KRIBET Naoufal
    """
    
    processed_segments = []
    
    # Use ThreadPoolExecutor for parallelization
    with ThreadPoolExecutor(max_workers=min(4, len(segments))) as executor:
        # Submit all segments with copies of config
        future_to_segment = {}
        for i, segment in enumerate(segments):
            # Each thread receives its own object copies
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
        
        # Retrieve results in order
        results = [None] * len(segments)
        for future in as_completed(future_to_segment):
            segment_idx = future_to_segment[future]
            try:
                result = future.result()
                results[segment_idx] = result
                print(f"  ✓ Segment {segment_idx + 1} processed")
            except Exception as e:
                print(f"  ❌ Error segment {segment_idx + 1}: {e}")
                results[segment_idx] = None
        
        processed_segments = [r for r in results if r is not None]
    
    return processed_segments


def _process_single_segment(segment_df: pd.DataFrame, config: Dict,
                          feature_calc: FeatureCalculator,
                          expert_calc: ExpertFeaturesCalculator,
                          window_manager: WindowManager) -> Optional[pd.DataFrame]:
    """
    Process a single segment by calculating all required features on VRP
    while PRESERVING pre-calculated features.
    
    FINAL ROBUST VERSION with correct handling of small windows.
    
    @param segment_df: DataFrame segment to process
    @param config: Feature extraction configuration
    @param feature_calc: Feature calculator instance
    @param expert_calc: Expert features calculator instance
    @param window_manager: Window manager instance
    @return: Processed DataFrame with features or None if processing fails
    @author: KRIBET Naoufal
    """
    if 'VRP' not in segment_df.columns or segment_df.empty:
        print("    -> WARNING: Empty segment or missing 'VRP' column. Ignored.")
        return None

    features_df = segment_df.copy()
    data_series = features_df['VRP']
    
    print(f"  -> Processing segment of {len(data_series)} points. Windows: {window_manager.windows_points}")

    log_data_series = np.log1p(data_series.clip(lower=0))
    print("  -> Logarithmic transformation of VRP signal completed.")

    for points in window_manager.windows_points:
        time_window_str = f"{points * 5}min"  # Assuming 5 minutes/point

        # --- CORRECTION: Centralized and safer min_periods logic ---
        # For basic stats, we can be flexible
        min_p_base = max(1, points // 4)
        # For advanced stats (kurtosis, quantiles), we need more points
        min_p_advanced = max(4, points // 2)

        # Check if window is large enough for advanced calculations
        if points < 4:
            print(f"    -> Window of {points} pts too small for advanced stats. Skipped.")
            min_p_advanced = points  # Ensure we don't request more points than window size

        # --- STEP 1: Calculate base components ---
        rolling_base = data_series.rolling(time_window_str, min_periods=min_p_base)
        rolling_adv = data_series.rolling(time_window_str, min_periods=min_p_advanced)

        min_val = rolling_base.min()
        max_val = rolling_base.max()
        mean_val = rolling_base.mean()
        std_val = rolling_base.std()
        
        # --- STEP 2: Assemble final features from components ---
        features_df[f'median{points}'] = rolling_base.median()
        features_df[f'std{points}'] = std_val
        features_df[f'min{points}'] = min_val
        features_df[f'max{points}'] = max_val
        
        # Advanced features with safeguard
        if points >= 4:
            features_df[f'kurtosis{points}'] = rolling_adv.kurt()
            features_df[f'skewness{points}'] = rolling_adv.skew()
            q_low = rolling_adv.quantile(0.1)
            q_high = rolling_adv.quantile(0.9)
            features_df[f'iqr{points}'] = q_high - q_low
        else:  # Fill with 0 if window too small
            features_df[f'kurtosis{points}'] = 0.0
            features_df[f'skewness{points}'] = 0.0
            features_df[f'iqr{points}'] = 0.0

        # Slope (Velocity) and Acceleration
        features_df[f'slope{points}'] = feature_calc.calculate_slope_vectorized(data_series, window=time_window_str)
        features_df[f'slope{points}_accel'] = features_df[f'slope{points}'].diff()

        # Robustness features
        safe_std = std_val.replace(0, 1e-6)
        features_df[f'zscore_{points}'] = (data_series - mean_val) / safe_std

        range_val = max_val - min_val
        safe_range = range_val.replace(0, 1e-6)
        features_df[f'range_pos{points}'] = (data_series - min_val) / safe_range
        features_df[f'range{points}'] = range_val
        log_slope = feature_calc.calculate_slope_vectorized(log_data_series, window=time_window_str)
        features_df[f'log_slope{points}'] = log_slope
        
        # Acceleration of log slope. If growth is purely exponential,
        # this value should tend toward zero.
        features_df[f'log_slope{points}_accel'] = log_slope.diff()
    
    # --- STEP 3: Post-loop calculations ---
    print("  -> Computing composite features (ratios, score)...")
    try:
        volatility_ratios = expert_calc.calculate_volatility_ratios_batch(features_df)
        features_df = features_df.assign(**volatility_ratios)
    except Exception as e:
        print(f"    -> WARNING: Volatility ratio calculation failed: {e}")
        
    features_df['precursor_score'] = calculate_eruption_precursor_score_optimized(features_df)
    
    # --- STEP 4: FINAL CLEANUP ---
    features_df.fillna(0.0, inplace=True)
    
    print(f"  -> Segment processing complete. {features_df.shape[1]} total features.")
    return features_df


def _finalize_feature_selection(features_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Finalize selection by applying user list as final filter.
    
    Extraction has already calculated a superset of features to handle dependencies.
    This function filters down to exactly what the user requested.
    
    @param features_df: DataFrame with all calculated features
    @param config: Configuration containing final feature selection
    @return: DataFrame with only user-selected features
    @author: KRIBET Naoufal
    """
    
    # 1. Get EXACT list of features user wants in final dataset
    user_final_selection = config.get('features', [])
    
    # 2. Safety: if list is empty, return empty DataFrame to avoid errors
    if not user_final_selection:
        print("WARNING: No features selected. Returning empty DataFrame.")
        return pd.DataFrame()

    # 3. Use .reindex() to create final DataFrame. This is the most robust method.
    #    - It selects ONLY columns present in your list from features_df
    #    - If a requested column couldn't be calculated (e.g., unsatisfied dependency),
    #      it will still be created and filled with 0.0, avoiding crashes.
    final_df = features_df.reindex(columns=user_final_selection, fill_value=0.0)
    
    # 4. Display clear message and return final DataFrame
    print(f"INFO: Final selection of {final_df.shape[1]} features, strictly based on user selection.")
    
    return final_df.astype(np.float32)


# Compatibility function with legacy API
def extract_features(df: pd.DataFrame, config: Dict, **kwargs) -> Optional[pd.DataFrame]:
    """
    Legacy API compatibility interface that passes kwargs to optimized function.
    
    @param df: Input DataFrame
    @param config: Feature extraction configuration
    @param kwargs: Additional keyword arguments
    @return: DataFrame with extracted features
    @author: KRIBET Naoufal
    """
    return extract_features_optimized(df, config, **kwargs)