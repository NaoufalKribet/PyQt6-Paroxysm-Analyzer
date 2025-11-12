"""
Data Processing Module for Seismic Event Analysis
==================================================

This module provides utilities for loading, cleaning, and preprocessing seismic data
for volcanic activity detection and event cycle analysis.

@author: KRIBET Naoufal
@affiliation: 5th year Engineering Student, EOST (École et Observatoire des Sciences de la Terre)
@date: 2025-11-12
@version: 1.0
"""

import pandas as pd
from typing import Optional, Dict, List
import numpy as np 

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads seismic data from Excel file with conservative cleaning approach.
    
    This function handles duplicate timestamps through aggregation, removes rows
    with missing critical data, and validates labels. No interpolation is performed
    to maintain data integrity.
    
    @param filepath: Path to the Excel file containing seismic data
    @return: Processed DataFrame or None if loading fails
    @raises FileNotFoundError: If the specified file does not exist
    @raises Exception: For any other unexpected errors during loading
    """
    try:
        df = pd.read_excel(filepath)
        print(f"File loaded: {filepath}. {len(df)} initial rows.")
    
        # Handle Date column and duplicate timestamps
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            if df.duplicated(subset=['Date']).any():
                num_duplicates = df['Date'].duplicated().sum()
                print(f"WARNING: {num_duplicates} duplicate timestamps detected. Aggregating by mean...")
                df.sort_values('Date', inplace=True)
                agg_rules = {col: 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'first' for col in df.columns if col != 'Date'}
                df = df.groupby('Date').agg(agg_rules).reset_index()
                print(f"  - Aggregation completed. DataFrame now contains {len(df)} unique rows.")

        # Clean missing values (NaN) by removal
        initial_rows = len(df)
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print("\nCleaning missing values (NaN) by removal...")
            print("NaN before cleaning:\n", nan_counts[nan_counts > 0])
            
            # Drop rows with NaN in critical columns
            critical_cols = [col for col in ['VRP', 'Ramp'] if col in df.columns]
            df.dropna(subset=critical_cols, inplace=True)
            
            rows_dropped = initial_rows - len(df)
            if rows_dropped > 0:
                print(f"  - {rows_dropped} rows were removed due to missing values in critical columns ('VRP', 'Ramp').")
            else:
                 print("  - No missing values in critical columns.")

        # Validate and normalize labels
        if 'Ramp' in df.columns:
            df['Ramp'] = df['Ramp'].astype(str).str.strip().str.title()
            print("\n--- Label Verification after cleaning ---")
            print("Unique labels:", df['Ramp'].unique().tolist())
            print("Label distribution:\n", df['Ramp'].value_counts())
            print("----------------------------------------------------------")
        
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {filepath}")
        return None
        
    except Exception as e:
        import traceback
        print(f"ERROR: An unexpected error occurred during loading. Trace: \n{traceback.format_exc()}")
        return None

def create_binary_target(df: pd.DataFrame, 
                         positive_class_name: str = 'Actif',
                         original_high_label: str = 'High',
                         target_col: str = 'Ramp') -> pd.DataFrame:
    """
    Creates a clean binary target variable from original multi-class labels.
    
    Transforms the original labeling scheme into a binary classification problem
    by mapping high activity events to the positive class and all other states
    to the negative class ('Calm').
    
    @param df: Input DataFrame with original labels
    @param positive_class_name: Name for the positive class (default: 'Actif')
    @param original_high_label: Original label for high activity events (default: 'High')
    @param target_col: Name of the target column (default: 'Ramp')
    @return: DataFrame with binary target labels
    
    @author: KRIBET Naoufal
    """
    print(f"--- Creating explicit binary target ('Calm' vs '{positive_class_name}') ---")
    df_new = df.copy()

    # Identify event rows
    is_event = (df_new[target_col].str.strip().str.title() == original_high_label)

    # Set all to 'Calm' first, then override events
    df_new[target_col] = 'Calm'
    df_new.loc[is_event, target_col] = positive_class_name
    
    print("New binary label distribution:")
    print(df_new[target_col].value_counts())
    print("-------------------------------------------------------------")
    
    return df_new

def define_internal_event_cycle(df_block: pd.DataFrame, 
                                pre_event_ratio: float, 
                                original_high_label: str = 'High',
                                target_col: str = 'Ramp') -> pd.DataFrame:
    """
    Defines internal phases within volcanic activity blocks.
    
    This function performs temporal phase labeling of seismic events based on
    relative position with respect to human-labeled 'High' activity markers.
    The methodology divides each activity block into three distinct phases:
    
    Phase Logic:
    1. Pre-Event: All data points before the first 'High' marker
    2. High-Paroxysm: The main event phase (subset of 'High' block)
    3. Post-Event: All data points after the last 'High' marker
    
    The 'High' block itself is subdivided using pre_event_ratio to separate
    the precursory phase from the paroxysmal phase.
    
    @param df_block: Block of activity data to be labeled
    @param pre_event_ratio: Ratio of 'High' block to label as 'Pre-Event' phase (0.0 to 1.0)
    @param original_high_label: Label marking high activity in original data (default: 'High')
    @param target_col: Name of the target column to be modified (default: 'Ramp')
    @return: DataFrame with phase labels (Pre-Event/High-Paroxysm/Post-Event)
    
    @author: KRIBET Naoufal
    """
    print(f"--- Defining event cycle (Pre/During/Post logic) ---")
    df = df_block.copy()
    
    # Find all 'High' indices
    high_indices = df.index[df[target_col].str.strip().str.title() == original_high_label]
    
    if high_indices.empty:
        print("  - WARNING: No 'High' found. Entire block labeled as 'Pre-Event'.")
        df[target_col] = 'Pre-Event'
        return df

    first_high_idx = high_indices.min()
    last_high_idx = high_indices.max()
    
    print(f"  - Anchoring on 'High' block from index {first_high_idx} to {last_high_idx}.")

    # Label everything before first 'High' as 'Pre-Event'
    df.loc[:first_high_idx - 1, target_col] = 'Pre-Event'

    # Label everything after last 'High' as 'Post-Event'
    df.loc[last_high_idx + 1:, target_col] = 'Post-Event'

    # Split the 'High' block into 'Pre-Event' and 'High-Paroxysm'
    event_duration_points = last_high_idx - first_high_idx + 1
    pre_duration_in_high = int(event_duration_points * pre_event_ratio)
    
    # Default to 'High-Paroxysm' for the entire 'High' block
    df.loc[first_high_idx:last_high_idx, target_col] = 'High-Paroxysm'

    # Override the beginning portion to 'Pre-Event'
    if pre_duration_in_high > 0:
        pre_event_end_idx = first_high_idx + pre_duration_in_high - 1
        df.loc[first_high_idx:pre_event_end_idx, target_col] = 'Pre-Event'

    print("\n  - Final label distribution for this block:")
    print(df[target_col].value_counts())
    print("-------------------------------------------------------------")
    
    return df

def balance_dataset(df: pd.DataFrame, params: Dict) -> Optional[pd.DataFrame]:
    """
    Balances dataset through event-based windowing strategy.
    
    This function creates a balanced dataset by extracting temporal windows around
    complete "High" activity events. Each window includes context before and after
    the event to provide the model with temporal evolution patterns.
    
    IMPORTANT NOTE: This function is designed for the legacy binary target definition
    ('High'/'Low'). It is NOT compatible with the refined multi-phase target
    ('Pre-Event'/'High-Paroxysm'/'Post-Event').
    
    @param df: Input DataFrame containing 'Ramp' column with event labels
    @param params: Dictionary containing balancing parameters:
                   - 'overrides': Dict mapping event indices to custom (start_offset, end_offset) tuples
    @return: Balanced DataFrame or None if error occurs
    @raises KeyError: If 'Ramp' column is not found
    
    """
    print("Starting data balancing process (event windowing method)...")
    
    if 'Ramp' not in df.columns:
        print("ERROR: 'Ramp' column not found. Balancing canceled.")
        return None

    # Identify 'High' events
    is_high = df['Ramp'] == "High"

    if not is_high.any():
        print("WARNING: No 'High' labels found. Windowing balance canceled.")
        return df

    # Detect event start and end points
    is_start_of_event = (is_high & ~is_high.shift(1, fill_value=False))
    is_end_of_event = (is_high & ~is_high.shift(-1, fill_value=False))

    index_high_first = df.index[is_start_of_event].tolist()
    index_high_last = df.index[is_end_of_event].tolist()
    
    num_events = len(index_high_first)
    print(f"Detected {num_events} complete 'High' events.")

    if num_events == 0:
        return df

    # Create balanced segments around each event
    balanced_segments = []
    overrides = params.get('overrides', {})

    for j, (first, last) in enumerate(zip(index_high_first, index_high_last), 1):
        start_offset, end_offset = 0, 0
        if j in overrides:
            start_offset, end_offset = overrides[j]
        else:
            # Default: symmetric window before and after event
            size_event = last - first + 1
            size_af = size_event // 2
            size_bef = size_event - size_af
            start_offset = -size_bef
            end_offset = size_af

        start_index = max(first + start_offset, 0)
        end_index = min(last + end_offset, len(df) - 1)
        balanced_segments.append(df.iloc[start_index : end_index + 1])

    if not balanced_segments:
        return df
        
    balanced_df = pd.concat(balanced_segments, ignore_index=True)
    print(f"Windowing balance completed. New DataFrame contains {len(balanced_df)} rows.")
    
    return balanced_df


def split_dataframe_on_gaps(df: pd.DataFrame, max_gap_duration: pd.Timedelta) -> List[pd.DataFrame]:
    """
    Segments time-series data based on temporal discontinuities.
    
    This function identifies large gaps in the temporal sequence and splits the
    DataFrame into continuous segments. This is crucial for volcanic monitoring
    where data acquisition may be interrupted due to equipment maintenance,
    transmission issues, or eruptive conditions.
    
    @param df: Input DataFrame with 'Date' column containing timestamps
    @param max_gap_duration: Maximum acceptable gap duration before splitting (pd.Timedelta)
    @return: List of DataFrame segments, each representing a continuous acquisition period
    @raises TypeError: If 'Date' column is missing from input DataFrame
    
    @note: Original temporal index is preserved for each segment
    """
    print(f"--- Searching for gaps larger than {max_gap_duration} for segmentation ---")
    
    df_copy = df.copy()

    if 'Date' not in df_copy.columns:
        raise TypeError("'Date' column is required for gap detection.")
    
    # Prepare temporal index
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True, drop=False)
    df_copy.sort_index(inplace=True)

    # Calculate time differences between consecutive points
    time_diffs = df_copy.index.to_series().diff()
    split_points_indices = time_diffs[time_diffs > max_gap_duration].index
    
    if len(split_points_indices) == 0:
        print("No major gaps detected. DataFrame will not be split.")
        return [df_copy]

    print(f"Detected {len(split_points_indices)} split points, creating {len(split_points_indices) + 1} segments.")
    
    # Create segments
    segments = []
    last_split_pos = 0
    
    split_positions = [df_copy.index.get_loc(idx) for idx in split_points_indices]

    for split_pos in split_positions:
        segment = df_copy.iloc[last_split_pos:split_pos]
        if not segment.empty:
            segments.append(segment)
        last_split_pos = split_pos

    # Add final segment
    last_segment = df_copy.iloc[last_split_pos:]
    if not last_segment.empty:
        segments.append(last_segment)
    
    print(f"DataFrame split into {len(segments)} non-empty segments.")
    return segments

def extract_activity_blocks(original_df: pd.DataFrame, detector_predictions: pd.Series) -> pd.DataFrame:
    """
    Extracts activity blocks from complete dataset based on detector predictions.
    
    This function implements a two-stage detection framework where a binary detector
    first identifies potential activity periods ('Actif'), and this function filters
    the original dataset to retain only those periods for subsequent detailed analysis.
    
    @param original_df: Complete dataset with 'Date' column
    @param detector_predictions: Series of binary predictions ('Calm'/'Actif') 
                                 with DatetimeIndex matching original_df timestamps
    @return: Filtered DataFrame containing only predicted active periods
    @raises ValueError: If 'Date' column is missing from original_df
    
    @note: This is a key component of the hierarchical detection pipeline
    @author: KRIBET Naoufal
    """
    if 'Date' not in original_df.columns:
        raise ValueError("Original DataFrame must contain 'Date' column.")
    
    # Set temporal index
    df = original_df.set_index('Date')
    
    # Extract dates where detector predicted activity
    active_dates = detector_predictions[detector_predictions == 'Actif'].index
    
    # Filter original data to active periods only
    active_df = df.reindex(active_dates).dropna(how='all')
    
    return active_df.reset_index()