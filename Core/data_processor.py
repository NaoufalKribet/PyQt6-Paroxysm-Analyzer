import pandas as pd
from typing import Optional, Dict, List
import numpy as np 

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Charge les données, gère les timestamps en double par agrégation, 
    supprime les lignes avec des données critiques manquantes, et valide les étiquettes.
    APPROCHE CONSERVATRICE SANS INTERPOLATION.
    """
    try:
        df = pd.read_excel(filepath)
        print(f"Fichier chargé : {filepath}. {len(df)} lignes initiales.")
    
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            if df.duplicated(subset=['Date']).any():
                num_duplicates = df['Date'].duplicated().sum()
                print(f"AVERTISSEMENT : {num_duplicates} timestamps en double détectés. Agrégation par moyenne...")
                df.sort_values('Date', inplace=True)
                agg_rules = {col: 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'first' for col in df.columns if col != 'Date'}
                df = df.groupby('Date').agg(agg_rules).reset_index()
                print(f"  - Agrégation terminée. Le DataFrame contient maintenant {len(df)} lignes uniques.")

        initial_rows = len(df)
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print("\nNettoyage des valeurs manquantes (NaN) par suppression...")
            print("NaN avant nettoyage :\n", nan_counts[nan_counts > 0])
            
            critical_cols = [col for col in ['VRP', 'Ramp'] if col in df.columns]
            df.dropna(subset=critical_cols, inplace=True)
            
            rows_dropped = initial_rows - len(df)
            if rows_dropped > 0:
                print(f"  - {rows_dropped} lignes ont été supprimées car elles avaient des valeurs manquantes dans les colonnes critiques ('VRP', 'Ramp').")
            else:
                 print("  - Aucune valeur manquante dans les colonnes critiques.")

        if 'Ramp' in df.columns:
            df['Ramp'] = df['Ramp'].astype(str).str.strip().str.title()
            print("\n--- Vérification des Étiquettes (Labels) après nettoyage ---")
            print("Étiquettes uniques :", df['Ramp'].unique().tolist())
            print("Distribution des étiquettes :\n", df['Ramp'].value_counts())
            print("----------------------------------------------------------")
        
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except FileNotFoundError:
        print(f"ERREUR : Le fichier n'a pas été trouvé à l'adresse : {filepath}")
        return None
        
    except Exception as e:
        import traceback
        print(f"ERREUR : Une erreur inattendue est survenue lors du chargement. Trace : \n{traceback.format_exc()}")
        return None

def create_binary_target(df: pd.DataFrame, 
                         positive_class_name: str = 'Actif',
                         original_high_label: str = 'High',
                         target_col: str = 'Ramp') -> pd.DataFrame:
    """
    Crée une cible binaire propre ('Calm' vs 'Actif') à partir des labels originaux.
    """
    print(f"--- Création d'une cible binaire explicite ('Calm' vs '{positive_class_name}') ---")
    df_new = df.copy()

    is_event = (df_new[target_col].str.strip().str.title() == original_high_label)

    df_new[target_col] = 'Calm'
    df_new.loc[is_event, target_col] = positive_class_name
    
    print("Nouvelle distribution binaire des étiquettes :")
    print(df_new[target_col].value_counts())
    print("-------------------------------------------------------------")
    
    return df_new
def define_internal_event_cycle(df_block: pd.DataFrame, 
                                pre_event_ratio: float, 
                                original_high_label: str = 'High',
                                target_col: str = 'Ramp') -> pd.DataFrame:
    """
    VERSION FINALE, DIRECTE ET ROBUSTE : Labellise les phases d'un bloc d'activité
    en se basant sur la position relative par rapport aux étiquettes 'High' humaines.
    
    La logique est la suivante :
    1. Tout ce qui précède le premier 'High' est 'Pre-Event'.
    2. Le bloc 'High' est scindé en 'Pre-Event' et 'High-Paroxysm' via un ratio.
    3. Tout ce qui suit le dernier 'High' est 'Post-Event'.
    """
    print(f"--- Définition du cycle d'événement (Logique Pré/Pendant/Post) ---")
    df = df_block.copy()
    
    high_indices = df.index[df[target_col].str.strip().str.title() == original_high_label]
    
    if high_indices.empty:
        print("  - AVERTISSEMENT: Aucun 'High' trouvé. Le bloc entier est labellisé 'Pre-Event'.")
        df[target_col] = 'Pre-Event'
        return df

    first_high_idx = high_indices.min()
    last_high_idx = high_indices.max()
    
    print(f"  - Ancrage sur le bloc 'High' allant de l'index {first_high_idx} à {last_high_idx}.")


    df.loc[:first_high_idx - 1, target_col] = 'Pre-Event'

    df.loc[last_high_idx + 1:, target_col] = 'Post-Event'

    event_duration_points = last_high_idx - first_high_idx + 1
    pre_duration_in_high = int(event_duration_points * pre_event_ratio)
    
    df.loc[first_high_idx:last_high_idx, target_col] = 'High-Paroxysm'

    if pre_duration_in_high > 0:
        pre_event_end_idx = first_high_idx + pre_duration_in_high - 1
        df.loc[first_high_idx:pre_event_end_idx, target_col] = 'Pre-Event'

    print("\n  - Distribution finale des labels pour ce bloc :")
    print(df[target_col].value_counts())
    print("-------------------------------------------------------------")
    
    return df
def balance_dataset(df: pd.DataFrame, params: Dict) -> Optional[pd.DataFrame]:
    """
    Équilibre le jeu de données en se basant sur les événements "High" complets.
    
    NOTE IMPORTANTE : Cette fonction est conçue pour l'ancienne définition de la cible
    ('High'/'Low'). Elle n'est PAS compatible avec la nouvelle cible 'Pre-Event'/'Stable'.
    """
    print("Début du processus d'équilibrage des données (méthode de fenêtrage d'événement)...")
    
    if 'Ramp' not in df.columns:
        print("ERREUR: La colonne 'Ramp' est introuvable. Équilibrage annulé.")
        return None

    is_high = df['Ramp'] == "High"

    if not is_high.any():
        print("AVERTISSEMENT: Aucune étiquette 'High' trouvée. L'équilibrage par fenêtrage est annulé.")
        return df

    is_start_of_event = (is_high & ~is_high.shift(1, fill_value=False))
    is_end_of_event = (is_high & ~is_high.shift(-1, fill_value=False))

    index_high_first = df.index[is_start_of_event].tolist()
    index_high_last = df.index[is_end_of_event].tolist()
    
    num_events = len(index_high_first)
    print(f"Détection de {num_events} événements 'High' complets.")

    if num_events == 0:
        return df

    balanced_segments = []
    overrides = params.get('overrides', {})

    for j, (first, last) in enumerate(zip(index_high_first, index_high_last), 1):
        start_offset, end_offset = 0, 0
        if j in overrides:
            start_offset, end_offset = overrides[j]
        else:
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
    print(f"Équilibrage par fenêtrage terminé. Le nouveau DataFrame contient {len(balanced_df)} lignes.")
    
    return balanced_df


def split_dataframe_on_gaps(df: pd.DataFrame, max_gap_duration: pd.Timedelta) -> List[pd.DataFrame]:
    """
    Scinde un DataFrame en une liste de segments basés sur les écarts de temps.
    Cette version corrigée préserve l'index temporel original pour chaque segment.
    """
    print(f"--- Recherche de gaps supérieurs à {max_gap_duration} pour la segmentation ---")
    
    df_copy = df.copy()

    if 'Date' not in df_copy.columns:
        raise TypeError("La colonne 'Date' est nécessaire pour la détection des gaps.")
    
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True, drop=False)
    df_copy.sort_index(inplace=True)

    time_diffs = df_copy.index.to_series().diff()
    split_points_indices = time_diffs[time_diffs > max_gap_duration].index
    
    if len(split_points_indices) == 0:
        print("Aucun gap majeur détecté. Le DataFrame ne sera pas scindé.")
        return [df_copy]

    print(f"Détection de {len(split_points_indices)} points de scission, créant {len(split_points_indices) + 1} segments.")
    segments = []
    last_split_pos = 0
    
    split_positions = [df_copy.index.get_loc(idx) for idx in split_points_indices]

    for split_pos in split_positions:
        segment = df_copy.iloc[last_split_pos:split_pos]
        if not segment.empty:
            segments.append(segment)
        last_split_pos = split_pos

    last_segment = df_copy.iloc[last_split_pos:]
    if not last_segment.empty:
        segments.append(last_segment)
    
    print(f"DataFrame scindé en {len(segments)} segments non vides.")
    return segments

def extract_activity_blocks(original_df: pd.DataFrame, detector_predictions: pd.Series) -> pd.DataFrame:
    """
    Filtre le DataFrame original pour ne conserver que les lignes
    où le Détecteur a prédit un état "Actif".

    Args:
        original_df (pd.DataFrame): Le jeu de données complet avec la colonne 'Date'.
        detector_predictions (pd.Series): La série des prédictions ('Calm'/'Actif')
                                           avec un DatetimeIndex correspondant.

    Returns:
        pd.DataFrame: Un nouveau DataFrame contenant uniquement les données des blocs actifs.
    """
    if 'Date' not in original_df.columns:
        raise ValueError("Le DataFrame original doit contenir la colonne 'Date'.")
    
    df = original_df.set_index('Date')
    
    active_dates = detector_predictions[detector_predictions == 'Actif'].index
    
    active_df = df.reindex(active_dates).dropna(how='all')
    

    return active_df.reset_index()
