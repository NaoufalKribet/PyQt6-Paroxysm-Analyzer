"""
Synthetic Seismic Data Generator for Volcanic Activity Simulation
==================================================================

This module generates realistic synthetic seismic datasets with rare volcanic events
of variable duration. The generated data mimics real volcanic monitoring scenarios
with different event types (short bursts and long eruptions) embedded in background
noise with realistic temporal patterns.

@author: KRIBET Naoufal
@affiliation: 5th year Engineering Student, EOST (École et Observatoire des Sciences de la Terre)
@date: 2025-11-12
@version: 1.0
"""

import pandas as pd
import numpy as np
import datetime

def generate_realistic_synthetic_data(
    num_points: int,
    num_events_total: int,
    ratio_long_events: float = 0.2,
    sampling_minutes: int = 5
) -> pd.DataFrame:
    """
    Generates highly realistic synthetic dataset with rare events of variable duration.
    
    This function creates a time-series dataset simulating volcanic seismic monitoring
    with the following characteristics:
    - Background signal with long-term drift and realistic noise
    - Two types of volcanic events: short bursts and long eruptions
    - Non-overlapping event placement with intelligent spacing
    - Pre-event, paroxysmal, and post-event phases for each event
    - Binary labels: 'High' (during paroxysm) and 'Low' (background/precursory/post)
    
    Event Types:
    - Short Burst: Hours-scale events with rapid onset and decay
    - Long Eruption: Days-scale events with extended precursory and post-eruptive phases
    
    @param num_points: Total number of time samples to generate
    @param num_events_total: Total number of volcanic events to insert
    @param ratio_long_events: Proportion of long eruptions vs short bursts (0.0 to 1.0)
    @param sampling_minutes: Time interval between samples in minutes
    @return: DataFrame with columns ['Date', 'VRP', 'Ramp']
             - Date: Timestamp for each sample
             - VRP: Volcanic Radiated Power (synthetic seismic amplitude)
             - Ramp: Binary label ('High' during paroxysm, 'Low' otherwise)
    
    @note: Events are randomly placed across the timeline without temporal overlap
    @note: VRP values are clipped to non-negative range to simulate physical reality
    @author: KRIBET Naoufal
    """
    
    # --- Configuration for two event types ---
    config = {
        'base': {
            'calm_level_mean': 150,           # Mean background activity level
            'calm_level_std': 20,             # Variability in background level
            'calm_noise_std': 15,             # Standard noise amplitude
            'long_term_period_days': 365,     # Period of seasonal drift (1 year)
        },
        'short_burst': {
            'amplitude_mean': 800,                    # Mean peak amplitude
            'amplitude_std': 200,                     # Amplitude variability
            'pre_duration_hours_mean': 4,             # Mean precursory phase duration
            'pre_duration_hours_std': 2,              # Precursory duration variability
            'high_duration_hours_mean': 6,            # Mean paroxysmal phase duration
            'high_duration_hours_std': 3,             # Paroxysmal duration variability
            'post_duration_hours_mean': 12,           # Mean post-event decay duration
            'post_duration_hours_std': 4,             # Post-event duration variability
        },
        'long_eruption': {
            'amplitude_mean': 1500,                   # Mean peak amplitude (higher than bursts)
            'amplitude_std': 400,                     # Amplitude variability
            'pre_duration_days_mean': 1.5,            # Mean precursory phase duration
            'pre_duration_days_std': 0.5,             # Precursory duration variability
            'high_duration_days_mean': 2.5,           # Mean paroxysmal phase duration
            'high_duration_days_std': 1,              # Paroxysmal duration variability
            'post_duration_days_mean': 3,             # Mean post-event decay duration
            'post_duration_days_std': 1,              # Post-event duration variability
        }
    }
    
    print("--- Generating realistic synthetic data... ---")
    
    # 1. Create temporal axis and background signal
    start_date = datetime.datetime.now()  # Start simulation from current time
    dates = pd.date_range(start=start_date, periods=num_points, freq=f'{sampling_minutes}min')
    
    # Generate background signal components
    base_cfg = config['base']
    calm_level = np.random.normal(base_cfg['calm_level_mean'], base_cfg['calm_level_std'])
    
    # Add long-term sinusoidal drift (seasonal variation)
    long_term_drift = 100 * np.sin(
        2 * np.pi * np.arange(num_points) / 
        ((base_cfg['long_term_period_days'] * 24 * 60) / sampling_minutes)
    )
    
    # Add random noise
    noise = np.random.normal(0, base_cfg['calm_noise_std'], num_points)
    
    # Combine background components
    vrp = calm_level + long_term_drift + noise
    labels = np.full(num_points, 'Low', dtype=object)
    
    # 2. Intelligent non-overlapping event placement
    num_long_events = int(num_events_total * ratio_long_events)
    num_short_events = num_events_total - num_long_events
    
    # Create shuffled list of event types to place
    event_types_to_place = ['long'] * num_long_events + ['short'] * num_short_events
    np.random.shuffle(event_types_to_place)
    
    placed_intervals = []  # Track placed event intervals to avoid overlaps
    MAX_TRIES = num_events_total * 50  # Maximum placement attempts per event
    
    # Place each event
    for event_type in event_types_to_place:
        is_event_placed = False
        
        for _ in range(MAX_TRIES):
            # Select configuration and generate random durations for this event
            if event_type == 'long':
                cfg = config['long_eruption']
                pre_d = max(1, np.random.normal(cfg['pre_duration_days_mean'], cfg['pre_duration_days_std'])) * 24 * 60
                high_d = max(1, np.random.normal(cfg['high_duration_days_mean'], cfg['high_duration_days_std'])) * 24 * 60
                post_d = max(1, np.random.normal(cfg['post_duration_days_mean'], cfg['post_duration_days_std'])) * 24 * 60
            else:  # short burst
                cfg = config['short_burst']
                pre_d = max(1, np.random.normal(cfg['pre_duration_hours_mean'], cfg['pre_duration_hours_std'])) * 60
                high_d = max(1, np.random.normal(cfg['high_duration_hours_mean'], cfg['high_duration_hours_std'])) * 60
                post_d = max(1, np.random.normal(cfg['post_duration_hours_mean'], cfg['post_duration_hours_std'])) * 60
            
            # Convert durations from minutes to sample points
            pre_duration = int(pre_d / sampling_minutes)
            high_duration = int(high_d / sampling_minutes)
            post_duration = int(post_d / sampling_minutes)
            total_duration = pre_duration + high_duration + post_duration
            
            # Try random placement
            start = np.random.randint(0, num_points - total_duration - 1)
            end = start + total_duration
            
            # Check for overlap with existing events
            is_overlapping = any(start < p_end and end > p_start for p_start, p_end in placed_intervals)
            
            if not is_overlapping:
                # Event can be placed - add to tracking list
                placed_intervals.append((start, end))
                
                # Generate event signal shape
                amplitude = np.random.normal(cfg['amplitude_mean'], cfg['amplitude_std'])
                
                # Pre-event: Quadratic ramp-up (accelerating precursors)
                pre_shape = amplitude * (np.linspace(0, 1, pre_duration)**2)
                
                # High paroxysm: Sustained high activity with noise
                high_shape = amplitude + np.random.normal(0, base_cfg['calm_noise_std'] * 4, high_duration)
                
                # Post-event: Exponential decay
                post_shape = amplitude * np.exp(-np.linspace(0, 5, post_duration))
                
                # Concatenate all phases
                full_event_shape = np.concatenate([pre_shape, high_shape, post_shape])
                
                # Add event to signal
                vrp[start:end] += full_event_shape
                
                # Label only the paroxysmal phase as 'High'
                labels[start + pre_duration : start + pre_duration + high_duration] = 'High'
                
                print(f"Event '{event_type}' placed (duration: {total_duration*sampling_minutes/60:.1f}h) at position {start}")
                is_event_placed = True
                break

        if not is_event_placed:
            print(f"WARNING: Unable to place event of type '{event_type}'.")

    # 3. Create output DataFrame
    df = pd.DataFrame({'Date': dates, 'VRP': vrp, 'Ramp': labels})
    
    # Clip VRP to non-negative values (physical constraint)
    df['VRP'] = df['VRP'].clip(lower=0)
    
    print("--- Generation completed. ---")
    return df