# Core/synthetic_data_generator.py

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
    Génère un jeu de données synthétique très réaliste avec des événements rares et de durées variables.
    Les événements sont placés aléatoirement sur toute la durée, sans se chevaucher.
    Le fichier de sortie ne contient que les labels 'High' et 'Low'.
    """
    
    # --- Configuration des deux types d'événements ---
    config = {
        'base': {
            'calm_level_mean': 150, 'calm_level_std': 20, 'calm_noise_std': 15,
            'long_term_period_days': 365,
        },
        'short_burst': {
            'amplitude_mean': 800, 'amplitude_std': 200,
            'pre_duration_hours_mean': 4, 'pre_duration_hours_std': 2,
            'high_duration_hours_mean': 6, 'high_duration_hours_std': 3,
            'post_duration_hours_mean': 12, 'post_duration_hours_std': 4,
        },
        'long_eruption': {
            'amplitude_mean': 1500, 'amplitude_std': 400,
            'pre_duration_days_mean': 1.5, 'pre_duration_days_std': 0.5,
            'high_duration_days_mean': 2.5, 'high_duration_days_std': 1,
            'post_duration_days_mean': 3, 'post_duration_days_std': 1,
        }
    }
    
    print("--- Génération de données synthétiques réalistes... ---")
    
    # 1. Création de l'axe temporel et du signal de base
    start_date = datetime.datetime.now() # Démarrer la simulation à partir de maintenant
    dates = pd.date_range(start=start_date, periods=num_points, freq=f'{sampling_minutes}min')
    
    base_cfg = config['base']
    calm_level = np.random.normal(base_cfg['calm_level_mean'], base_cfg['calm_level_std'])
    long_term_drift = 100 * np.sin(2 * np.pi * np.arange(num_points) / ((base_cfg['long_term_period_days'] * 24 * 60) / sampling_minutes))
    noise = np.random.normal(0, base_cfg['calm_noise_std'], num_points)
    
    vrp = calm_level + long_term_drift + noise
    labels = np.full(num_points, 'Low', dtype=object)
    
    # 2. Placement intelligent et non-chevauchant des événements
    num_long_events = int(num_events_total * ratio_long_events)
    num_short_events = num_events_total - num_long_events
    
    event_types_to_place = ['long'] * num_long_events + ['short'] * num_short_events
    np.random.shuffle(event_types_to_place)
    
    placed_intervals = []
    MAX_TRIES = num_events_total * 50
    
    for event_type in event_types_to_place:
        is_event_placed = False
        for _ in range(MAX_TRIES):
            if event_type == 'long':
                cfg = config['long_eruption']
                pre_d = max(1, np.random.normal(cfg['pre_duration_days_mean'], cfg['pre_duration_days_std'])) * 24 * 60
                high_d = max(1, np.random.normal(cfg['high_duration_days_mean'], cfg['high_duration_days_std'])) * 24 * 60
                post_d = max(1, np.random.normal(cfg['post_duration_days_mean'], cfg['post_duration_days_std'])) * 24 * 60
            else:
                cfg = config['short_burst']
                pre_d = max(1, np.random.normal(cfg['pre_duration_hours_mean'], cfg['pre_duration_hours_std'])) * 60
                high_d = max(1, np.random.normal(cfg['high_duration_hours_mean'], cfg['high_duration_hours_std'])) * 60
                post_d = max(1, np.random.normal(cfg['post_duration_hours_mean'], cfg['post_duration_hours_std'])) * 60
            
            pre_duration = int(pre_d / sampling_minutes); high_duration = int(high_d / sampling_minutes); post_duration = int(post_d / sampling_minutes)
            total_duration = pre_duration + high_duration + post_duration
            
            start = np.random.randint(0, num_points - total_duration - 1)
            end = start + total_duration
            
            is_overlapping = any(start < p_end and end > p_start for p_start, p_end in placed_intervals)
            
            if not is_overlapping:
                placed_intervals.append((start, end))
                amplitude = np.random.normal(cfg['amplitude_mean'], cfg['amplitude_std'])
                pre_shape = amplitude * (np.linspace(0, 1, pre_duration)**2)
                high_shape = amplitude + np.random.normal(0, base_cfg['calm_noise_std'] * 4, high_duration)
                post_shape = amplitude * np.exp(-np.linspace(0, 5, post_duration))
                full_event_shape = np.concatenate([pre_shape, high_shape, post_shape])
                
                vrp[start:end] += full_event_shape
                labels[start + pre_duration : start + pre_duration + high_duration] = 'High'
                
                print(f"Événement '{event_type}' placé (durée: {total_duration*sampling_minutes/60:.1f}h) à la position {start}")
                is_event_placed = True
                break

        if not is_event_placed:
            print(f"AVERTISSEMENT: Impossible de placer un événement de type '{event_type}'.")

    df = pd.DataFrame({'Date': dates, 'VRP': vrp, 'Ramp': labels})
    df['VRP'] = df['VRP'].clip(lower=0)
    
    print("--- Génération terminée. ---")
    return df