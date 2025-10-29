# Core/model_trainer.py (Version Complète, Corrigée pour sample_weight)

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.stats import randint
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import time
from sklearn.metrics import matthews_corrcoef 
# Au début de Core/model_trainer.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import make_scorer 



def _get_positive_class_f1_scorer(y_train: pd.Series):
    """
    Crée un scorer scikit-learn qui optimise le F1-score
    spécifiquement pour la classe "positive" (celle qui n'est pas 'Calm').
    
    Cette fonction s'adapte au contexte binaire ('Actif') ou multi-classe ('Pre-Event').
    """
    labels_order = sorted(y_train.unique())
    
    # Stratégie de détection de la classe positive
    # Priorité 1 : La classe la plus critique est 'Pre-Event'
    if 'Pre-Event' in labels_order:
        positive_class_name = 'Pre-Event'
    # Priorité 2 : Sinon, c'est la classe qui n'est pas 'Calm'
    else:
        positive_class_name = next((label for label in labels_order if label != 'Calm'), None)

    # Si on a bien trouvé une classe positive à cibler
    if positive_class_name:
        try:
            positive_class_pos = labels_order.index(positive_class_name)
            
            def positive_class_f1_func(y_true, y_pred):
                # Calculer les F1-scores pour toutes les classes dans l'ordre
                f1_scores = f1_score(y_true, y_pred, average=None, labels=labels_order, zero_division=0)
                # Retourner le score de notre classe cible
                return f1_scores[positive_class_pos]
            
            custom_scorer = make_scorer(positive_class_f1_func)
            logging.info(f"Métrique d'optimisation (scoring) : F1-Score de la classe CIBLE '{positive_class_name}'.")
            return custom_scorer

        except (ValueError, IndexError):
            # Sécurité si quelque chose se passe mal
            pass

    # Cas par défaut si aucune classe positive n'est trouvée
    logging.warning("Impossible de déterminer une classe positive unique. Optimisation sur 'f1_macro'.")
    return 'f1_macro'

# Dans Core/model_trainer.py

def _get_f1_macro_scorer():
    """
    Retourne le nom du scorer pour le F1-Score Macro.

    Cette métrique est choisie pour l'optimisation des hyperparamètres car elle
    offre le meilleur équilibre pour les problèmes de classification déséquilibrés.
    Elle calcule le F1-Score pour chaque classe indépendamment, puis en fait la
    moyenne simple, donnant ainsi un poids égal à la performance sur la classe
    minoritaire ('Actif') et sur la classe majoritaire ('Calm').

    Ceci force le modèle à trouver un compromis robuste, en étant bon à la fois
    pour détecter les événements (bon Rappel pour 'Actif') et pour ne pas générer
    trop de fausses alarmes (bonne Précision pour 'Actif', ce qui implique
    un bon Rappel pour 'Calm').

    Returns:
        str: Le nom du scorer 'f1_macro' pour scikit-learn.
    """
    logging.info("Métrique d'optimisation (scoring) : F1-Score Macro (équilibré).")
    
    # Pour les métriques standards de scikit-learn, il suffit de retourner leur nom.
    # RandomizedSearchCV saura l'interpréter correctement.
    return 'f1_macro'

@dataclass
class ModelConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    n_iter: int = 20
    cv_folds: int = 3
    max_k: int = 30
    weights: list = None
    metrics: list = None
    max_trees: int = 200
    max_depth: int = 15
    min_leaf: int = 1
    use_class_weight: bool = False
    nn_epochs: int = 200
    nn_batch_size: int = 64
    nn_learning_rate: float = 0.001
    # --- FIN DE L'AJOUT ---
    
    def __post_init__(self):
        if self.weights is None: self.weights = ['uniform', 'distance']
        if self.metrics is None: self.metrics = ['minkowski']

class ModelTrainerError(Exception):
    """Exception personnalisée pour les erreurs d'entraînement."""
    pass

@contextmanager
def timing_context(operation_name: str):
    """Context manager pour mesurer le temps d'exécution."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logging.info(f"{operation_name} terminé en {elapsed_time:.2f}s")


# --- Entraînement KNN (inchangé) ---

def train_and_evaluate_knn(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          config: ModelConfig) -> Dict:
    with timing_context("Recherche d'hyperparamètres KNN"):
        best_params, _ = _find_best_knn_params(X_train, y_train, X_val, y_val, config)
    
    with timing_context("Entraînement du modèle KNN final"):
        X_combined = pd.concat([X_train, X_val], axis=0); y_combined = pd.concat([y_train, y_val], axis=0)
        final_knn = KNeighborsClassifier(**best_params).fit(X_combined, y_combined)
    
    with timing_context("Évaluation finale KNN"):
        results = _evaluate_model(final_knn, X_test, y_test, best_params)
    
    return results

def _find_best_knn_params(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         config: ModelConfig) -> Tuple[Dict, float]:
    best_params = {}; best_f1 = -1.0
    param_grid = [{'n_neighbors': k, 'weights': w, 'metric': m} for k in range(1, config.max_k + 1) for w in config.weights for m in config.metrics]
    for params in param_grid:
        try:
            knn = KNeighborsClassifier(**params).fit(X_train, y_train)
            y_val_pred = knn.predict(X_val)
            current_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            if current_f1 > best_f1: best_f1 = current_f1; best_params = params
        except Exception as e: logging.warning(f"Erreur avec les paramètres KNN {params}: {str(e)}")
    if not best_params: raise ModelTrainerError("Aucun paramètre valide trouvé pour KNN")
    logging.info(f"Meilleurs paramètres KNN trouvés sur validation : {best_params} (F1={best_f1:.4f})")
    return best_params, best_f1


# --- Entraînement Random Forest (CORRIGÉ POUR ACCEPTER sample_weight) ---

def train_and_evaluate_rf(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                         X_test: pd.DataFrame, y_test: pd.Series,
                         config: ModelConfig,
                         sample_weight: Optional[np.ndarray] = None) -> Dict:
    with timing_context("Recherche d'hyperparamètres Random Forest (Temporelle)"):
        best_rf, best_cv_score = _find_best_rf_model(X_train, y_train, X_val, y_val, config, sample_weight)
    
    with timing_context("Évaluation finale Random Forest"):
        results = _evaluate_model(best_rf, X_test, y_test, best_rf.get_params())
    
    with timing_context("Analyse de généralisation (Validation Set)"):
        logging.info("--- Analyse de Généralisation (Diagnostic) ---")
        logging.info(f"Performance sur la validation croisée (interne): F1-macro = {best_cv_score:.4f}")
        test_score = results['report']['macro avg']['f1-score']
        logging.info(f"Performance sur le jeu de TEST (jamais vu): F1-macro = {test_score:.4f}")
        gap = abs(test_score - best_cv_score)
        logging.info(f"Écart Validation/Test : {gap:.4f}. Un faible écart est un signe de bonne généralisation.")
        logging.info("--- Fin de l'Analyse ---")
    return results

# Dans Core/model_trainer.py

def _find_best_rf_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                       config: ModelConfig,
                       sample_weight: Optional[np.ndarray] = None) -> tuple:
    
    logging.info("--- Lancement de la recherche RF - Stratégie: Alerte Précoce ---")

    # --- CORRECTION : Définition dynamique des poids de classe ---
    class_weights = None
    if config.use_class_weight:
        unique_labels = y_train.unique()
        
        # Cas du Détecteur Binaire
        if len(unique_labels) <= 2:
            # Trouver dynamiquement le nom de la classe positive (celle qui n'est pas 'Calm')
            positive_class = [label for label in unique_labels if label != 'Calm']
            
            if len(positive_class) == 1:
                positive_class_name = positive_class[0]
                # Construire le dictionnaire dynamiquement
                class_weights = {'Calm': 1, positive_class_name: 5}
                logging.info(f"Mode Binaire détecté. Poids de classe dynamiques appliqués : {class_weights}")
            else:
                # Sécurité : si on ne trouve pas une config claire, on utilise la méthode par défaut
                logging.warning("Impossible de déterminer la classe positive unique. Utilisation de 'balanced'.")
                class_weights = 'balanced'
        
        # Cas du Segmenteur Multi-classes (reste inchangé)
        else:
            class_weights = {
                'Calm': 1,
                'Pre-Event': 15,
                'High-Paroxysm': 10,
                'Post-Event': 5
            }
            logging.info(f"Mode Multi-classes détecté. Poids de classe appliqués : {class_weights}")
            
    # --- Fin de la correction ---

    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(8, 25),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
        # On passe le dictionnaire de poids qui vient d'être créé
        'class_weight': [class_weights]
    }
    
    # Le reste de la fonction est inchangé...
    X_combined = pd.concat([X_train, X_val]) if y_val is not None else X_train
    y_combined = pd.concat([y_train, y_val]) if y_val is not None else y_train
    
    rf_base = RandomForestClassifier(random_state=config.random_state, n_jobs=-1)
    time_series_cv = TimeSeriesSplit(n_splits=config.cv_folds)
    
    fit_params = {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight
        logging.info("Utilisation de la pondération temporelle (sample_weight) pour l'entraînement RF.")

    custom_scorer = _get_f1_macro_scorer()

    random_search = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=param_distributions, 
        n_iter=config.n_iter,
        cv=time_series_cv, 
        scoring=custom_scorer,
        n_jobs=-1, 
        random_state=config.random_state, 
        error_score=0, 
        verbose=1
    )
    
    try:
        random_search.fit(X_combined, y_combined, **fit_params)
        best_rf, best_score = random_search.best_estimator_, random_search.best_score_
        logging.info(f"Meilleurs paramètres RF : {random_search.best_params_}")
        logging.info(f"Meilleur score (CV) sur la métrique ciblée : {best_score:.4f}")
        return best_rf, best_score
    except Exception as e:
        raise ModelTrainerError(f"Erreur lors de la recherche Random Forest : {str(e)}")
    
from sklearn.metrics import f1_score, make_scorer
import logging

# ... (le reste de vos imports et fonctions) ...

# Dans Core/model_trainer.py (remplacez la fonction existante par celle-ci)

# Assurez-vous d'avoir ces imports en haut de votre fichier model_trainer.py
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import randint
import lightgbm as lgb
import logging
# ... (et les autres imports comme pandas, numpy, RandomizedSearchCV, etc.)


def train_and_evaluate_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                            X_test: pd.DataFrame, y_test: pd.Series,
                            config: ModelConfig,
                            sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Entraîne et évalue un modèle LightGBM avec une stratégie agressive de
    détection de précurseurs (Pre-Event).

    Cette fonction est spécifiquement conçue pour résoudre le problème de la
    non-détection des classes minoritaires critiques. Elle utilise :
    1. Une métrique de scoring personnalisée pour que la recherche d'hyperparamètres
       se concentre EXCLUSIVEMENT sur l'amélioration du F1-Score de la classe 'Pre-Event'.
    2. Des poids de classe manuels et très asymétriques pour pénaliser lourdement
       les erreurs sur les classes 'Pre-Event' et 'High-Paroxysm'.
    """
    logging.info("--- Lancement de l'entraînement LGBM - Stratégie: Alerte Précoce Agressive ---")

    # --- Étape 1 : Définir la stratégie de pondération des classes ---
    # Nous prenons le contrôle manuel pour sur-pénaliser les erreurs sur les précurseurs.
    class_weights = None
    if config.use_class_weight:
        class_weights = {
            'Calm': 1,
            'Pre-Event': 50,      # Priorité n°1 : La détection est 50x plus importante que le calme.
            'High-Paroxysm': 30,  # Priorité n°2 : Détecter le pic est critique, mais moins que l'alerte.
            'Post-Event': 10      # Priorité n°3 : Utile pour le cycle, mais moins critique.
        }
        logging.info(f"Utilisation de poids de classe manuels et ciblés : {class_weights}")

    # --- Étape 2 : Créer une métrique d'optimisation personnalisée ---
    # Le but n'est plus le 'f1_macro', mais le 'f1' de la classe 'Pre-Event'.
    # Note: La fonction _get_pre_event_f1_scorer doit exister dans votre fichier.
    custom_scorer = _get_f1_macro_scorer()
    
    # --- Étape 3 : Configuration et lancement de la recherche d'hyperparamètres ---
    with timing_context("Recherche d'hyperparamètres LightGBM (Ciblée Pre-Event)"):
        # Espace de recherche large pour trouver les meilleurs paramètres
        param_distributions = {
            'n_estimators': randint(200, 1500),
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': randint(20, 60),
            'max_depth': randint(5, 20),
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        }
        
        # Modèle de base avec les poids de classe injectés
        lgbm_base = lgb.LGBMClassifier(
            random_state=config.random_state,
            n_jobs=-1,
            class_weight=class_weights
        )
        
        # Combiner les données d'entraînement et de validation pour la cross-validation
        X_combined = pd.concat([X_train, X_val]) if y_val is not None else X_train
        y_combined = pd.concat([y_train, y_val]) if y_val is not None else y_train
        
        # Utiliser une cross-validation temporelle
        time_series_cv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        # Préparer les paramètres d'ajustement (pour les poids temporels)
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            logging.info("Utilisation de la pondération temporelle (sample_weight) pour l'entraînement LGBM.")

        # L'outil de recherche
        random_search = RandomizedSearchCV(
            estimator=lgbm_base,
            param_distributions=param_distributions,
            n_iter=config.n_iter,  # Augmenter n_iter dans l'UI (ex: 30 ou 50) donnera de meilleurs résultats
            cv=time_series_cv,
            scoring=custom_scorer, # <--- LA MODIFICATION LA PLUS IMPORTANTE
            n_jobs=-1,
            random_state=config.random_state,
            error_score=0,
            verbose=2 # Affiche plus de détails pendant la recherche
        )
        
        try:
            # Lancer la recherche
            random_search.fit(X_combined, y_combined, **fit_params)
            best_lgbm, best_cv_score = random_search.best_estimator_, random_search.best_score_
            logging.info(f"Meilleurs hyperparamètres trouvés (optimisés pour Pre-Event F1) : {random_search.best_params_}")
            logging.info(f"Meilleur score F1 'Pre-Event' (Validation Croisée) : {best_cv_score:.4f}")
        except Exception as e:
            raise ModelTrainerError(f"Erreur critique lors de la recherche LightGBM : {str(e)}")

    # --- Étape 4 : Évaluation finale sur le jeu de test jamais vu ---
    with timing_context("Évaluation finale LightGBM sur le jeu de test"):
        # Note: La fonction _evaluate_model doit exister dans votre fichier.
        results = _evaluate_model(best_lgbm, X_test, y_test, best_lgbm.get_params())
    
    # --- Étape 5 : Analyse de la performance et de la généralisation ---
    logging.info("--- ANALYSE DE LA PERFORMANCE D'ALERTE PRÉCOCE ---")
    report = results.get('report', {})
    
    # Extraire les F1-scores des classes critiques du rapport final
    pre_event_f1_test = report.get('Pre-Event', {}).get('f1-score', 0.0)
    high_f1_test = report.get('High-Paroxysm', {}).get('f1-score', 0.0)
    
    logging.info(f"Performance sur Validation Croisée (F1 Pre-Event) : {best_cv_score:.4f}")
    logging.info(f"Performance sur JEU DE TEST (F1 Pre-Event) : {pre_event_f1_test:.4f}")
    logging.info(f"Performance sur JEU DE TEST (F1 High-Paroxysm) : {high_f1_test:.4f}")
    
    if pre_event_f1_test > 0.1: # Seuil de succès réaliste
        logging.info("SUCCÈS : Le modèle a réussi à identifier des signaux 'Pre-Event' !")
    else:
        logging.error("AVERTISSEMENT : Le modèle a encore des difficultés à prédire la classe 'Pre-Event'.")

    # Vérifier si le modèle n'a pas trop "sur-appris"
    gap = abs(pre_event_f1_test - best_cv_score)
    logging.info(f"Écart de généralisation (Validation/Test) sur Pre-Event F1 : {gap:.4f}. Un faible écart est un signe de bonne généralisation.")
    logging.info("-----------------------------------------------------")

    return results
def _evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   best_params: Dict) -> Dict:
    """
    Évalue le modèle final sur le jeu de test.
    VERSION MISE À JOUR : Capture et retourne les probabilités de chaque classe.
    """
    try:
        # --- MODIFICATION : CAPTURE DES PROBABILITÉS ---
        y_test_pred_proba = model.predict_proba(X_test)
        # --- FIN DE LA MODIFICATION ---
        
        y_test_pred_raw = model.predict(X_test)
        y_test_pred_series = pd.Series(y_test_pred_raw, index=X_test.index)
        
        logging.info("Application du filtre de persistance temporelle sur les prédictions...")
        y_test_pred_filtered = y_test_pred_series.copy()
        min_persistence_steps = 3
        
        positive_classes = [c for c in y_test_pred_series.unique() if c != 'Calm']
        
        for p_class in positive_classes:
            is_positive_class = (y_test_pred_series == p_class)
            if is_positive_class.any():
                blocks = (is_positive_class.diff() != 0).cumsum()
                block_counts = is_positive_class.groupby(blocks).transform('sum')
                alerts_to_cancel = is_positive_class & (block_counts < min_persistence_steps)
                y_test_pred_filtered[alerts_to_cancel] = 'Calm'
                logging.info(f"{alerts_to_cancel.sum()} points isolés de la classe '{p_class}' ont été supprimés.")
        
        y_test_pred = y_test_pred_filtered.values
        class_labels = model.classes_ # Utiliser l'ordre des classes du modèle est plus sûr
        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0, labels=class_labels)
        cm = confusion_matrix(y_test, y_test_pred, labels=class_labels)
        mcc = matthews_corrcoef(y_test, y_test_pred)
        
        results = {
            'model': model,
            'best_params': best_params,
            'predictions': y_test_pred,
            # --- MODIFICATION : AJOUT DES PROBABILITÉS AUX RÉSULTATS ---
            'probabilities': y_test_pred_proba,
            # --- FIN DE LA MODIFICATION ---
            'report': report,
            'confusion_matrix': cm,
            'class_names': class_labels.tolist(),
            'mcc': mcc
        }
        
        # --- CORRECTION : LOGGING CONTEXTUALISÉ ---
        logging.info(f"--- Résultats Finaux sur le Jeu de Test (après filtrage) ---")
        
        # S'il s'agit du Détecteur Binaire, on rapporte la performance sur 'Actif'
        if 'Actif' in report:
            target_report = report.get('Actif', {})
            target_class_name = 'Actif'
        # Sinon (pour le Segmenteur), on rapporte la performance sur 'Pre-Event'
        elif 'Pre-Event' in report:
            target_report = report.get('Pre-Event', {})
            target_class_name = 'Pre-Event'
        # Cas par défaut
        else:
            target_report = {}
            target_class_name = 'N/A'

        recall = target_report.get('recall', 0)
        precision = target_report.get('precision', 0)
        macro_f1 = report.get('macro avg', {}).get('f1-score', 0)
        
        logging.info(f"Rappel '{target_class_name}'    : {recall:.4f}")
        logging.info(f"Précision '{target_class_name}' : {precision:.4f}")
        logging.info(f"F1-Score Macro        : {macro_f1:.4f}")
        logging.info(f"Score MCC             : {mcc:.4f} (Score entre -1 et 1, 0 = aléatoire)")
        logging.info("-----------------------------------------------------------")

        # Le reste de la fonction (importance des features) est inchangé...
        if hasattr(model, 'feature_importances_'):
            logging.info("--- Importance Complète des Caractéristiques (Classement Console) ---")
            importances_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
            for index, row in importances_df.iterrows():
                logging.info(f"  {index + 1:2d}. {row['Feature']:<35} | Score: {row['Importance']:.6f}")
            logging.info("-----------------------------------------------------------------")
        
        return results
        
    except Exception as e:
        import traceback
        model_name = type(model).__name__
        raise ModelTrainerError(f"Erreur évaluation finale ({model_name}): {traceback.format_exc()}")

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [ModelTrainer] - %(message)s')

# À la fin de Core/model_trainer.py

# Dans Core/model_trainer.py, REMPLACEZ cette fonction

def train_and_evaluate_nn(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                          X_test: pd.DataFrame, y_test: pd.Series,
                          config: ModelConfig,
                          sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Entraîne un classifieur NN, évalue sa performance et retourne un dictionnaire complet
    de résultats, incluant les probabilités pour chaque classe.
    """
    logging.info("--- Lancement de l'entraînement avec un Réseau de Neurones (Classifieur) ---")

    # --- Étape 1 : Préparation des données (encodage des labels) ---
    all_labels = np.unique(np.concatenate([y_train, y_val if y_val is not None else [], y_test]))
    label_encoder = LabelEncoder().fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    
    onehot_encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(len(all_labels))])
    y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
    
    validation_data = None
    if y_val is not None and not y_val.empty:
        y_val_encoded = label_encoder.transform(y_val)
        y_val_onehot = onehot_encoder.transform(y_val_encoded.reshape(-1, 1))
        validation_data = (X_val, y_val_onehot)

    # --- Étape 2 : Définition de l'architecture et compilation ---
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_train_onehot.shape[1], activation='softmax')
    ])
    
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=config.nn_learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info(model.summary())

    # --- Étape 3 : Entraînement ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    with timing_context("Entraînement du réseau de neurones"):
        history = model.fit(X_train, y_train_onehot,
                            epochs=config.nn_epochs,
                            batch_size=config.nn_batch_size,
                            validation_data=validation_data,
                            callbacks=[early_stopping],
                            verbose=2)

    # --- Étape 4 : Évaluation finale sur le jeu de test ---
    with timing_context("Évaluation finale du NN"):
        # --- LA CORRECTION EST ICI : ON CAPTURE LES PROBABILITÉS ---
        y_pred_proba = model.predict(X_test)
        # --- FIN DE LA CORRECTION ---
        
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Appliquer le filtre de persistance (identique à _evaluate_model)
        y_pred_series = pd.Series(y_pred, index=X_test.index)
        y_pred_filtered = y_pred_series.copy()
        positive_classes = [c for c in y_pred_series.unique() if c != 'Calm']
        for p_class in positive_classes:
            is_positive = (y_pred_series == p_class)
            if is_positive.any():
                blocks = (is_positive.diff() != 0).cumsum()
                counts = is_positive.groupby(blocks).transform('sum')
                alerts_to_cancel = is_positive & (counts < 3)
                y_pred_filtered[alerts_to_cancel] = 'Calm'
        
        y_pred_final = y_pred_filtered.values
        
        # Calcul des métriques
        report = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_final, labels=label_encoder.classes_)
        
        # --- ON AJOUTE LES PROBABILITÉS AU DICTIONNAIRE DE RÉSULTATS ---
        results = {
            'model': model,
            'best_params': {'epochs': config.nn_epochs, 'batch_size': config.nn_batch_size, 'lr': config.nn_learning_rate},
            'predictions': y_pred_final,
            'probabilities': y_pred_proba, # <-- LA LIGNE AJOUTÉE
            'report': report,
            'confusion_matrix': cm,
            'class_names': label_encoder.classes_.tolist(),
            'training_history': history.history
        }
    
    return results

# ==============================================================================
# == NOUVELLE SECTION : LSTM GUIDÉ PAR LA PHYSIQUE (PINN)                      ==
# ==============================================================================
from tensorflow.keras.layers import LSTM, Input
import tensorflow as tf

def create_sequences(data: pd.DataFrame, sequence_length: int, target_col: str = 'VRP'):
    """
    Transforme une série temporelle en un jeu de données supervisé pour les LSTM.
    Entrée (X): une séquence de `sequence_length` points passés.
    Sortie (y): le point suivant.
    """
    X, y, indices = [], [], []
    for i in range(len(data) - sequence_length):
        X.append(data[target_col].iloc[i:(i + sequence_length)].values)
        y.append(data[target_col].iloc[i + sequence_length])
        indices.append(data.index[i + sequence_length])

    # Redimensionner pour Keras: [samples, timesteps, features]
    return np.array(X).reshape(-1, sequence_length, 1), np.array(y), indices

def create_forecasting_model(sequence_length: int):
    """Crée un modèle LSTM simple pour la prévision."""
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(50, return_sequences=True),
        LSTM(30, return_sequences=False),
        Dense(20, activation='relu'),
        Dense(1) # Couche de sortie pour prédire la prochaine valeur VRP
    ])
    return model

def derive_state_from_forecast(y_true_vrp, y_pred_vrp, calm_threshold=300, paroxysm_threshold=1000, growth_threshold=0.05):
    """
    Traduit les prédictions de VRP d'un modèle de forecasting en labels de classe.
    """
    labels = []
    # On commence à l'indice 1 car on a besoin de y_true_vrp[i-1]
    for i in range(1, len(y_pred_vrp)):
        current_vrp = y_true_vrp[i-1]
        predicted_vrp = y_pred_vrp[i]
        
        if current_vrp > paroxysm_threshold and predicted_vrp < current_vrp:
            labels.append('Post-Event')
        elif predicted_vrp > paroxysm_threshold:
            labels.append('High-Paroxysm')
        elif predicted_vrp > current_vrp * (1 + growth_threshold) and current_vrp > calm_threshold:
             labels.append('Pre-Event')
        else:
            labels.append('Calm')
            
    return np.array(['Calm'] + labels)

def train_and_evaluate_lstm_pinn(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 config: ModelConfig,
                                 sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Entraîne un modèle LSTM de prévision avec une fonction de coût guidée par la physique.
    """
    logging.info("--- Lancement de l'entraînement LSTM Guidé par la Physique (PINN) ---")

    # --- 1. Préparation des données en séquences ---
    sequence_length = 50 
    
    train_df = pd.DataFrame({'VRP': y_train.values}, index=y_train.index)
    val_df = pd.DataFrame({'VRP': y_val.values}, index=y_val.index)
    test_df = pd.DataFrame({'VRP': y_test.values}, index=y_test.index)
    
    X_train_seq, y_train_seq, _ = create_sequences(train_df, sequence_length)
    X_val_seq, y_val_seq, _ = create_sequences(val_df, sequence_length)
    X_test_seq, y_test_seq, test_indices = create_sequences(test_df, sequence_length)

    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))
    X_val_scaled = scaler_X.transform(X_val_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y_val_scaled = scaler_y.transform(y_val_seq.reshape(-1, 1))

    # --- 2. Création du modèle et des outils d'entraînement ---
    model = create_forecasting_model(sequence_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.nn_learning_rate)
    
    def physics_loss(y_true_batch, y_pred_batch, last_point_in_sequence, lambda_mono=0.5):
        mse_loss = tf.keras.losses.mean_squared_error(y_true_batch, y_pred_batch)
        monotonicity_error = tf.maximum(0., last_point_in_sequence - y_pred_batch)
        monotonicity_loss = tf.reduce_mean(monotonicity_error)
        return tf.reduce_mean(mse_loss) + lambda_mono * monotonicity_loss

    # --- 3. Boucle d'entraînement personnalisée ---
    epochs = config.nn_epochs
    batch_size = config.nn_batch_size
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []} # Pour compatibilité

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = []
        for i in range(0, len(X_train_scaled), batch_size):
            X_batch = X_train_scaled[i:i+batch_size]; y_batch = y_train_scaled[i:i+batch_size]
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                last_points = X_batch[:, -1, :]
                loss = physics_loss(y_batch, y_pred, last_points)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss.numpy())
        
        val_preds = model(X_val_scaled, training=False)
        last_points_val = X_val_scaled[:, -1, :]
        val_loss = physics_loss(y_val_scaled, val_preds, last_points_val)
        
        history['loss'].append(np.mean(epoch_loss))
        history['val_loss'].append(val_loss.numpy())
        print(f"  loss: {np.mean(epoch_loss):.4f} - val_loss: {val_loss.numpy():.4f}")

    # --- 4. Évaluation avec interprétation ---
    X_test_scaled = scaler_X.transform(X_test_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred_vrp = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_pred_labels = derive_state_from_forecast(y_test_seq, y_pred_vrp)
    
    from .data_processor import define_internal_event_cycle
    original_test_labels = y_test.iloc[sequence_length:].reset_index(drop=True)
    y_true_labels_df = pd.DataFrame({'VRP': y_test_seq, 'Ramp': original_test_labels})
    y_true_labels_df_processed = define_internal_event_cycle(y_true_labels_df, pre_event_ratio=0.3)
    y_true_labels = y_true_labels_df_processed['Ramp'].values
    
    unique_labels = np.unique(np.concatenate((y_true_labels, y_pred_labels)))
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0, labels=unique_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)
    
    return {
        'model': model, 'scalers': {'X': scaler_X, 'y': scaler_y},
        'best_params': {'sequence_length': sequence_length},
        'predictions': pd.Series(y_pred_labels, index=test_indices),
        'report': report, 'confusion_matrix': cm, 'class_names': unique_labels.tolist(),
        'training_history': history
    }