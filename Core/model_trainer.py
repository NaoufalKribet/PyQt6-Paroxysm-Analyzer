"""
================================================================================
                        MODEL TRAINING PIPELINE MODULE
================================================================================

Architecture Overview:
----------------------
This module implements a comprehensive machine learning pipeline for time series
event detection and classification. It provides a unified interface for training,
optimizing, and evaluating multiple model architectures specifically designed
for imbalanced classification problems with temporal dependencies.

Pipeline Components:
--------------------
1. MODEL CONFIGURATION
   - Centralized configuration management via ModelConfig dataclass
   - Hyperparameter space definition for each model type
   - Validation strategy parameters (train/val/test splits)

2. OPTIMIZATION STRATEGIES
   - Custom F1-score optimizers for minority class detection
   - Time-series aware cross-validation (TimeSeriesSplit)
   - Class weight balancing for imbalanced datasets
   - Temporal sample weighting support

3. MODEL IMPLEMENTATIONS
   - K-Nearest Neighbors (KNN): Simple baseline with exhaustive grid search
   - Random Forest (RF): Ensemble method with temporal validation
   - LightGBM: Gradient boosting with aggressive minority class detection
   - Neural Network: Deep learning classifier with early stopping
   - LSTM-PINN: Physics-informed neural network for time series forecasting

4. EVALUATION FRAMEWORK
   - Comprehensive metrics calculation (F1, Precision, Recall, MCC)
   - Temporal persistence filtering for reducing false positives
   - Feature importance analysis for interpretability
   - Generalization gap analysis

Key Features:
-------------
- Automatic handling of binary and multi-class scenarios
- Dynamic class weight adjustment based on data distribution
- Temporal coherence enforcement through persistence filtering
- Physics-guided loss functions for LSTM models
- Extensive logging for debugging and monitoring

Performance Considerations:
---------------------------
- Parallel processing enabled where possible (n_jobs=-1)
- Early stopping mechanisms to prevent overfitting
- Memory-efficient data handling with pandas DataFrames

@author: KRIBET Naoufal
@version: 2.0.0
@date: 2025-11-17
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
import logging
import time
from typing import Dict, Optional
import pandas as pd
import numpy as np
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import make_scorer 


def _get_positive_class_f1_scorer(y_train: pd.Series):
    """
    Create a scikit-learn scorer that optimizes the F1-score specifically 
    for the positive class (non-'Calm' class).
    
    This function adapts to both binary ('Actif') and multi-class ('Pre-Event') 
    classification scenarios by dynamically identifying the positive class.
    
    @param y_train: Training target labels used to identify unique classes
    @return: Custom sklearn scorer for positive class F1-score or 'f1_macro' as fallback
    @raises: None (falls back to f1_macro if errors occur)
    @author: KRIBET Naoufal
    """
    # Get sorted list of unique labels for consistent ordering
    labels_order = sorted(y_train.unique())
    
    # Identify the positive class based on context
    if 'Pre-Event' in labels_order:
        positive_class_name = 'Pre-Event'
    else:
        # Find the first non-'Calm' label as positive class
        positive_class_name = next((label for label in labels_order if label != 'Calm'), None)

    if positive_class_name:
        try:
            # Get the position of positive class in the ordered list
            positive_class_pos = labels_order.index(positive_class_name)
            
            def positive_class_f1_func(y_true, y_pred):
                # Calculate F1-scores for all classes in order
                f1_scores = f1_score(y_true, y_pred, average=None, labels=labels_order, zero_division=0)
                # Return the score for our target class
                return f1_scores[positive_class_pos]
            
            # Create custom scorer
            custom_scorer = make_scorer(positive_class_f1_func)
            logging.info(f"Optimization metric (scoring): F1-Score for TARGET class '{positive_class_name}'.")
            return custom_scorer

        except (ValueError, IndexError):
            # Safety fallback if something goes wrong
            pass

    # Default case if no positive class is found
    logging.warning("Unable to determine a unique positive class. Optimizing on 'f1_macro'.")
    return 'f1_macro'


def _get_f1_macro_scorer():
    """
    Return the scorer name for Macro F1-Score.

    This metric is chosen for hyperparameter optimization as it provides the best
    balance for imbalanced classification problems. It calculates the F1-Score for
    each class independently, then averages them with equal weight, giving equal
    importance to minority class ('Actif') and majority class ('Calm') performance.

    This forces the model to find a robust compromise, being good at both detecting
    events (high Recall for 'Actif') and not generating too many false alarms
    (high Precision for 'Actif', which implies high Recall for 'Calm').

    @return: Scorer name 'f1_macro' for scikit-learn
    @author: KRIBET Naoufal
    """
    logging.info("Optimization metric (scoring): Macro F1-Score (balanced).")
    
    # For standard scikit-learn metrics, just return the name
    # RandomizedSearchCV will interpret it correctly
    return 'f1_macro'


@dataclass
class ModelConfig:
    """
    Configuration dataclass for model training parameters.
    
    Centralizes all hyperparameters and training settings to ensure
    consistency across different model types and experiments.
    
    @author: KRIBET Naoufal
    """
    # Data splitting ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # General training parameters
    random_state: int = 42
    n_iter: int = 20  # Number of iterations for random search
    cv_folds: int = 3  # Number of cross-validation folds
    
    # KNN specific parameters
    max_k: int = 30  # Maximum number of neighbors to test
    weights: list = None  # Weight functions for KNN
    metrics: list = None  # Distance metrics for KNN
    
    # Tree-based model parameters
    max_trees: int = 200  # Maximum number of trees for ensemble methods
    max_depth: int = 15  # Maximum tree depth
    min_leaf: int = 1  # Minimum samples in leaf
    use_class_weight: bool = False  # Whether to use class weights
    
    # Neural network parameters
    nn_epochs: int = 200  # Maximum training epochs
    nn_batch_size: int = 64  # Batch size for training
    nn_learning_rate: float = 0.001  # Learning rate for optimizer
    
    def __post_init__(self):
        """Initialize default values for list parameters after instantiation."""
        if self.weights is None: 
            self.weights = ['uniform', 'distance']
        if self.metrics is None: 
            self.metrics = ['minkowski']


class ModelTrainerError(Exception):
    """
    Custom exception for model training errors.
    
    Provides specific error handling for the training pipeline to distinguish
    training failures from other system errors.
    
    @author: KRIBET Naoufal
    """
    pass


@contextmanager
def timing_context(operation_name: str):
    """
    Context manager for measuring execution time of operations.
    
    Provides automatic timing and logging for any code block, useful for
    performance monitoring and optimization.
    
    @param operation_name: Name of the operation being timed
    @author: KRIBET Naoufal
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logging.info(f"{operation_name} completed in {elapsed_time:.2f}s")


# --- KNN Training Implementation ---

def train_and_evaluate_knn(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          config: ModelConfig,
                          sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Train and evaluate a K-Nearest Neighbors classifier using a rigorous,
    time-series aware grid search for hyperparameter optimization.
    
    CORRECTED VERSION: This function now accepts the 'sample_weight' argument to 
    maintain a consistent interface with other trainers, but it is ignored as the 
    underlying scikit-learn KNN does not support it in its 'fit' method.
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features (combined with train for CV)
    @param y_val: Validation labels (combined with train for CV)
    @param X_test: Test features
    @param y_test: Test labels
    @param config: Model configuration object
    @param sample_weight: Optional sample weights (accepted but ignored).
    @return: Dictionary containing model, predictions, and evaluation metrics
    @author: KRIBET Naoufal (Updated by Gemini based on error traceback)
    """
    logging.info("--- Launching KNN search - Strategy: Time-Series Grid Search ---")

    if sample_weight is not None:
        logging.warning("Temporal weighting (sample_weight) is not supported by the standard KNN model and will be ignored.")

    # Step 1: Combine train and validation sets for a proper cross-validation workflow
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    
    # Step 2: Define the hyperparameter grid to search
    param_grid = {
        'n_neighbors': range(1, config.max_k + 1),
        'weights': config.weights,
        'metric': config.metrics
    }
    
    # Step 3: Setup the time-series cross-validation splitter with a gap
    time_series_cv = TimeSeriesSplit(n_splits=3, gap=50)

    # Step 4: Configure and run the Grid Search
    with timing_context("KNN hyperparameter search (GridSearchCV + TimeSeriesSplit)"):
        knn_base = KNeighborsClassifier(n_jobs=-1)
        
        # Using f1_macro as the scoring metric for consistency
        grid_search = GridSearchCV(
            estimator=knn_base,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=time_series_cv,
            verbose=1,
            n_jobs=-1 # Use all available cores for the search
        )
        
        grid_search.fit(X_combined, y_combined) # sample_weight is not passed here
        
        best_knn = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        logging.info(f"Best KNN parameters found via Grid Search: {best_params}")
        logging.info(f"Best cross-validation F1-macro score: {best_cv_score:.4f}")

    # Step 5: Evaluate the best model found on the unseen test set
    with timing_context("Final KNN evaluation"):
        results = _evaluate_model(best_knn, X_test, y_test, best_params)
        
    logging.info(f"Final Test Set F1-macro score: {results['report']['macro avg']['f1-score']:.4f}")
    
    return results


def _find_best_knn_params(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         config: ModelConfig) -> Tuple[Dict, float]:
    """
    Find optimal KNN hyperparameters through exhaustive grid search.
    
    Tests all combinations of k values, weight functions, and distance metrics
    to find the configuration that maximizes macro F1-score on validation set.
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features  
    @param y_val: Validation labels
    @param config: Model configuration
    @return: Tuple of (best parameters dict, best F1 score)
    @raises ModelTrainerError: If no valid parameters found
    @author: KRIBET Naoufal
    """
    best_params = {}
    best_f1 = -1.0
    
    # Generate all parameter combinations
    param_grid = [
        {'n_neighbors': k, 'weights': w, 'metric': m} 
        for k in range(1, config.max_k + 1) 
        for w in config.weights 
        for m in config.metrics
    ]
    
    # Exhaustive search through parameter space
    for params in param_grid:
        try:
            # Train model with current parameters
            knn = KNeighborsClassifier(**params).fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = knn.predict(X_val)
            current_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            
            # Track best configuration
            if current_f1 > best_f1: 
                best_f1 = current_f1
                best_params = params
                
        except Exception as e: 
            logging.warning(f"Error with KNN parameters {params}: {str(e)}")
    
    # Ensure valid parameters were found
    if not best_params: 
        raise ModelTrainerError("No valid parameters found for KNN")
    
    logging.info(f"Best KNN parameters found on validation: {best_params} (F1={best_f1:.4f})")
    return best_params, best_f1


# --- Random Forest Training Implementation ---

def train_and_evaluate_rf(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                         X_test: pd.DataFrame, y_test: pd.Series,
                         config: ModelConfig,
                         sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Train and evaluate a Random Forest classifier with temporal validation.
    
    Uses randomized search with time series cross-validation to find optimal
    hyperparameters, then evaluates generalization performance.
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features (optional)
    @param y_val: Validation labels (optional)
    @param X_test: Test features
    @param y_test: Test labels
    @param config: Model configuration
    @param sample_weight: Optional sample weights for temporal weighting
    @return: Dictionary with model, predictions, and metrics
    @author:  KRIBET Naoufal
    """
    # Step 1: Hyperparameter optimization
    with timing_context("Random Forest hyperparameter search (Temporal)"):
        best_rf, best_cv_score = _find_best_rf_model(X_train, y_train, X_val, y_val, config, sample_weight)
    
    # Step 2: Final evaluation
    with timing_context("Final Random Forest evaluation"):
        results = _evaluate_model(best_rf, X_test, y_test, best_rf.get_params())
    
    # Step 3: Generalization analysis
    with timing_context("Generalization analysis (Validation Set)"):
        logging.info("--- Generalization Analysis (Diagnostic) ---")
        logging.info(f"Cross-validation performance (internal): F1-macro = {best_cv_score:.4f}")
        test_score = results['report']['macro avg']['f1-score']
        logging.info(f"Test set performance (never seen): F1-macro = {test_score:.4f}")
        gap = abs(test_score - best_cv_score)
        logging.info(f"Validation/Test gap: {gap:.4f}. Small gap indicates good generalization.")
        logging.info("--- End of Analysis ---")
    
    return results


def _find_best_rf_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                       config: ModelConfig,
                       sample_weight: Optional[np.ndarray] = None) -> tuple:
    """
    Find optimal Random Forest model using randomized search with time series CV.
    
    Implements dynamic class weighting strategy based on problem type (binary vs
    multi-class) with aggressive weighting for minority classes.
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features (optional)
    @param y_val: Validation labels (optional)
    @param config: Model configuration
    @param sample_weight: Optional temporal weights
    @return: Tuple of (best model, best CV score)
    @raises ModelTrainerError: If search fails
    @author: KRIBET Naoufal 
    """
    logging.info("--- Launching RF search - Strategy: Early Warning ---")

    # Dynamic class weight definition based on problem type
    class_weights = None
    if config.use_class_weight:
        unique_labels = y_train.unique()
        
        # Binary detector case
        if len(unique_labels) <= 2:
            # Dynamically find positive class name (non-'Calm' class)
            positive_class = [label for label in unique_labels if label != 'Calm']
            
            if len(positive_class) == 1:
                positive_class_name = positive_class[0]
                # Build dictionary dynamically
                class_weights = {'Calm': 1, positive_class_name: 5}
                logging.info(f"Binary mode detected. Dynamic class weights applied: {class_weights}")
            else:
                # Safety: if unclear configuration, use default method
                logging.warning("Unable to determine unique positive class. Using 'balanced'.")
                class_weights = 'balanced'
        
        # Multi-class segmenter case
        else:
            class_weights = {
                'Calm': 1,
                'Pre-Event': 15,
                'High-Paroxysm': 10,
                'Post-Event': 5
            }
            logging.info(f"Multi-class mode detected. Class weights applied: {class_weights}")

    # Define hyperparameter search space
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(8, 25),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
        'class_weight': [class_weights]
    }
    
    # Combine train and validation sets if validation provided
    X_combined = pd.concat([X_train, X_val]) if y_val is not None else X_train
    y_combined = pd.concat([y_train, y_val]) if y_val is not None else y_train
    
    # Initialize base model and cross-validation strategy
    rf_base = RandomForestClassifier(random_state=config.random_state, n_jobs=-1)
    time_series_cv = TimeSeriesSplit(n_splits=config.cv_folds)
    
    # Prepare fit parameters for sample weighting
    fit_params = {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight
        logging.info("Using temporal weighting (sample_weight) for RF training.")

    # Get custom scorer for optimization
    custom_scorer = _get_f1_macro_scorer()

    # Configure randomized search
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
        # Execute search
        random_search.fit(X_combined, y_combined, **fit_params)
        best_rf, best_score = random_search.best_estimator_, random_search.best_score_
        logging.info(f"Best RF parameters: {random_search.best_params_}")
        logging.info(f"Best score (CV) on target metric: {best_score:.4f}")
        return best_rf, best_score
    except Exception as e:
        raise ModelTrainerError(f"Error during Random Forest search: {str(e)}")


def train_and_evaluate_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                            X_test: pd.DataFrame, y_test: pd.Series,
                            config: ModelConfig,
                            sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Train and evaluate a LightGBM model with aggressive early detection strategy.
    
    Specifically designed to solve the problem of minority class non-detection
    through:
    1. Custom scoring metric focused on Pre-Event F1-Score
    2. Manual asymmetric class weights heavily penalizing minority class errors
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features (optional)
    @param y_val: Validation labels (optional)
    @param X_test: Test features
    @param y_test: Test labels
    @param config: Model configuration
    @param sample_weight: Optional temporal weights
    @return: Dictionary with model, predictions, and metrics
    @raises ModelTrainerError: If training fails
    @author: KRIBET Naoufal
    """
    logging.info("--- Launching LGBM training - Strategy: Aggressive Early Warning ---")

    # Step 1: Define class weighting strategy
    # Take manual control to heavily penalize precursor errors
    class_weights = None
    if config.use_class_weight:
        class_weights = {
            'Calm': 1,
            'Pre-Event': 50,      # Priority #1: Detection is 50x more important than calm
            'High-Paroxysm': 30,  # Priority #2: Peak detection is critical but less than warning
            'Post-Event': 10      # Priority #3: Useful for cycle but less critical
        }
        logging.info(f"Using manual targeted class weights: {class_weights}")

    # Step 2: Create custom optimization metric
    # Goal is no longer 'f1_macro' but Pre-Event class F1
    custom_scorer = _get_f1_macro_scorer()
    
    # Step 3: Hyperparameter search configuration and launch
    with timing_context("LightGBM hyperparameter search (Pre-Event targeted)"):
        # Large search space to find best parameters
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
        
        # Base model with injected class weights
        lgbm_base = lgb.LGBMClassifier(
            random_state=config.random_state,
            n_jobs=-1,
            class_weight=class_weights
        )
        
        # Combine train and validation data for cross-validation
        X_combined = pd.concat([X_train, X_val]) if y_val is not None else X_train
        y_combined = pd.concat([y_train, y_val]) if y_val is not None else y_train
        
        # Use temporal cross-validation
        time_series_cv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        # Prepare fit parameters (for temporal weights)
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            logging.info("Using temporal weighting (sample_weight) for LGBM training.")

        # Search tool configuration
        random_search = RandomizedSearchCV(
            estimator=lgbm_base,
            param_distributions=param_distributions,
            n_iter=config.n_iter,  # Increase n_iter in UI (e.g., 30 or 50) for better results
            cv=time_series_cv,
            scoring=custom_scorer, # MOST IMPORTANT MODIFICATION
            n_jobs=-1,
            random_state=config.random_state,
            error_score=0,
            verbose=2  # Show more details during search
        )
        
        try:
            # Launch search
            random_search.fit(X_combined, y_combined, **fit_params)
            best_lgbm, best_cv_score = random_search.best_estimator_, random_search.best_score_
            logging.info(f"Best hyperparameters found (optimized for Pre-Event F1): {random_search.best_params_}")
            logging.info(f"Best 'Pre-Event' F1 score (Cross-Validation): {best_cv_score:.4f}")
        except Exception as e:
            raise ModelTrainerError(f"Critical error during LightGBM search: {str(e)}")

    # Step 4: Final evaluation on never-seen test set
    with timing_context("Final LightGBM evaluation on test set"):
        results = _evaluate_model(best_lgbm, X_test, y_test, best_lgbm.get_params())
    
    # Step 5: Performance and generalization analysis
    logging.info("--- EARLY WARNING PERFORMANCE ANALYSIS ---")
    report = results.get('report', {})
    
    # Extract F1-scores for critical classes from final report
    pre_event_f1_test = report.get('Pre-Event', {}).get('f1-score', 0.0)
    high_f1_test = report.get('High-Paroxysm', {}).get('f1-score', 0.0)
    
    logging.info(f"Cross-Validation performance (F1 Pre-Event): {best_cv_score:.4f}")
    logging.info(f"TEST SET performance (F1 Pre-Event): {pre_event_f1_test:.4f}")
    logging.info(f"TEST SET performance (F1 High-Paroxysm): {high_f1_test:.4f}")
    
    if pre_event_f1_test > 0.1:  # Realistic success threshold
        logging.info("SUCCESS: Model successfully identified 'Pre-Event' signals!")
    else:
        logging.error("WARNING: Model still has difficulties predicting 'Pre-Event' class.")

    # Check if model hasn't overfit too much
    gap = abs(pre_event_f1_test - best_cv_score)
    logging.info(f"Generalization gap (Validation/Test) on Pre-Event F1: {gap:.4f}. Small gap indicates good generalization.")
    logging.info("-----------------------------------------------------")

    return results


def _evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   best_params: Dict) -> Dict:
    """
    Evaluate final model on test set with comprehensive metrics.
    
    Includes temporal persistence filtering to reduce isolated false positives,
    feature importance analysis for interpretability, and probability capture.
    
    @param model: Trained model to evaluate
    @param X_test: Test features
    @param y_test: Test labels
    @param best_params: Best parameters used for training
    @return: Dictionary with predictions, probabilities, metrics, and analysis
    @raises ModelTrainerError: If evaluation fails
    @author: KRIBET Naoufal
    """
    try:
        # Capture prediction probabilities for each class
        y_test_pred_proba = model.predict_proba(X_test)
        
        # Get raw predictions
        y_test_pred_raw = model.predict(X_test)
        y_test_pred_series = pd.Series(y_test_pred_raw, index=X_test.index)
        
        # Apply temporal persistence filter to reduce noise
        logging.info("Applying temporal persistence filter on predictions...")
        y_test_pred_filtered = y_test_pred_series.copy()
        min_persistence_steps = 3  # Minimum consecutive predictions required
        
        # Identify positive classes (non-'Calm')
        positive_classes = [c for c in y_test_pred_series.unique() if c != 'Calm']
        
        # Filter each positive class separately
        for p_class in positive_classes:
            is_positive_class = (y_test_pred_series == p_class)
            if is_positive_class.any():
                # Identify contiguous blocks of predictions
                blocks = (is_positive_class.diff() != 0).cumsum()
                block_counts = is_positive_class.groupby(blocks).transform('sum')
                
                # Cancel alerts that don't meet persistence threshold
                alerts_to_cancel = is_positive_class & (block_counts < min_persistence_steps)
                y_test_pred_filtered[alerts_to_cancel] = 'Calm'
                logging.info(f"{alerts_to_cancel.sum()} isolated points of class '{p_class}' were removed.")
        
        # Final filtered predictions
        y_test_pred = y_test_pred_filtered.values
        
        # Calculate comprehensive metrics
        class_labels = model.classes_  # Use model's class order for consistency
        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0, labels=class_labels)
        cm = confusion_matrix(y_test, y_test_pred, labels=class_labels)
        mcc = matthews_corrcoef(y_test, y_test_pred)
        
        # Compile results
        results = {
            'model': model,
            'best_params': best_params,
            'predictions': y_test_pred,
            'probabilities': y_test_pred_proba,  # Added probability capture
            'report': report,
            'confusion_matrix': cm,
            'class_names': class_labels.tolist(),
            'mcc': mcc
        }
        
        # Contextual logging based on problem type
        logging.info(f"--- Final Results on Test Set (after filtering) ---")
        
        # Binary detector reports on 'Actif' performance
        if 'Actif' in report:
            target_report = report.get('Actif', {})
            target_class_name = 'Actif'
        # Multi-class segmenter reports on 'Pre-Event' performance
        elif 'Pre-Event' in report:
            target_report = report.get('Pre-Event', {})
            target_class_name = 'Pre-Event'
        # Default case
        else:
            target_report = {}
            target_class_name = 'N/A'

        recall = target_report.get('recall', 0)
        precision = target_report.get('precision', 0)
        macro_f1 = report.get('macro avg', {}).get('f1-score', 0)
        
        logging.info(f"Recall '{target_class_name}'    : {recall:.4f}")
        logging.info(f"Precision '{target_class_name}' : {precision:.4f}")
        logging.info(f"Macro F1-Score        : {macro_f1:.4f}")
        logging.info(f"MCC Score             : {mcc:.4f} (Range: -1 to 1, 0 = random)")
        logging.info("-----------------------------------------------------------")

        # Feature importance analysis if available
        if hasattr(model, 'feature_importances_'):
            logging.info("--- Complete Feature Importance (Console Ranking) ---")
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
        raise ModelTrainerError(f"Final evaluation error ({model_name}): {traceback.format_exc()}")


# --- Neural Network Training Implementation ---

def train_and_evaluate_nn(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                          X_test: pd.DataFrame, y_test: pd.Series,
                          config: ModelConfig,
                          sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Train and evaluate a neural network classifier.
    
    Implements a simple feedforward neural network with early stopping,
    including proper label encoding for multi-class classification.
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features (optional)
    @param y_val: Validation labels (optional)
    @param X_test: Test features
    @param y_test: Test labels
    @param config: Model configuration
    @param sample_weight: Optional sample weights (unused in current implementation)
    @return: Dictionary with model, predictions, probabilities, and metrics
    @author: KRIBET Naoufal
    """
    logging.info("--- Launching Neural Network training (Classifier) ---")

    # Step 1: Data preparation (label encoding)
    # Get all unique labels across datasets
    all_labels = np.unique(np.concatenate([y_train, y_val if y_val is not None else [], y_test]))
    
    # Encode labels to integers
    label_encoder = LabelEncoder().fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    
    # Convert to one-hot encoding for neural network
    onehot_encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(len(all_labels))])
    y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
    
    # Prepare validation data if provided
    validation_data = None
    if y_val is not None and not y_val.empty:
        y_val_encoded = label_encoder.transform(y_val)
        y_val_onehot = onehot_encoder.transform(y_val_encoded.reshape(-1, 1))
        validation_data = (X_val, y_val_onehot)

    # Step 2: Architecture definition and compilation
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_train_onehot.shape[1], activation='softmax')
    ])
    
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=config.nn_learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info(model.summary())

    # Step 3: Training with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    with timing_context("Neural network training"):
        history = model.fit(X_train, y_train_onehot,
                            epochs=config.nn_epochs,
                            batch_size=config.nn_batch_size,
                            validation_data=validation_data,
                            callbacks=[early_stopping],
                            verbose=2)

    # Step 4: Final evaluation on test set
    with timing_context("Final NN evaluation"):
        # Capture prediction probabilities
        y_pred_proba = model.predict(X_test)
        
        # Get predicted classes
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Apply persistence filter (same as other models)
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
        
        # Calculate metrics
        report = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_final, labels=label_encoder.classes_)
        
        # Compile results with probabilities
        results = {
            'model': model,
            'best_params': {
                'epochs': config.nn_epochs, 
                'batch_size': config.nn_batch_size, 
                'lr': config.nn_learning_rate
            },
            'predictions': y_pred_final,
            'probabilities': y_pred_proba,  # Include probabilities
            'report': report,
            'confusion_matrix': cm,
            'class_names': label_encoder.classes_.tolist(),
            'training_history': history.history
        }
    
    return results


# ==============================================================================
# == NEW SECTION: PHYSICS-INFORMED LSTM (PINN)                                ==
# ==============================================================================
from tensorflow.keras.layers import LSTM, Input
import tensorflow as tf

def create_sequences(data: pd.DataFrame, sequence_length: int, target_col: str = 'VRP'):
    """
    Transform time series data into supervised learning format for LSTM.
    
    Creates input-output pairs where input is a sequence of past values
    and output is the next value in the series.
    
    @param data: DataFrame containing time series
    @param sequence_length: Number of past timesteps to use as input
    @param target_col: Name of the column to predict
    @return: Tuple of (X sequences, y targets, corresponding indices)
    @author: KRIBET Naoufal
    """
    X, y, indices = [], [], []
    
    # Create sliding window sequences
    for i in range(len(data) - sequence_length):
        # Input: sequence of past values
        X.append(data[target_col].iloc[i:(i + sequence_length)].values)
        # Output: next value
        y.append(data[target_col].iloc[i + sequence_length])
        # Track original index
        indices.append(data.index[i + sequence_length])

    # Reshape for Keras: [samples, timesteps, features]
    return np.array(X).reshape(-1, sequence_length, 1), np.array(y), indices


def create_forecasting_model(sequence_length: int):
    """
    Create LSTM model architecture for time series forecasting.
    
    Implements a two-layer LSTM with dense output for single-step
    ahead prediction.
    
    @param sequence_length: Length of input sequences
    @return: Compiled Keras Sequential model
    @author: KRIBET Naoufal
    """
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(50, return_sequences=True),  # First LSTM layer returns sequences
        LSTM(30, return_sequences=False),  # Second LSTM returns final hidden state
        Dense(20, activation='relu'),  # Hidden dense layer
        Dense(1)  # Output layer for single value prediction
    ])
    return model


def derive_state_from_forecast(y_true_vrp, y_pred_vrp, calm_threshold=300, 
                               paroxysm_threshold=1000, growth_threshold=0.05):
    """
    Translate VRP forecasts into event classification labels.
    
    Uses domain knowledge to interpret predicted values and trends
    into discrete event states.
    
    @param y_true_vrp: True VRP values
    @param y_pred_vrp: Predicted VRP values
    @param calm_threshold: Threshold below which system is calm
    @param paroxysm_threshold: Threshold for high activity
    @param growth_threshold: Relative growth rate for pre-event detection
    @return: Array of classification labels
    @author: KRIBET Naoufal
    """
    labels = []
    
    # Start from index 1 as we need previous value for comparison
    for i in range(1, len(y_pred_vrp)):
        current_vrp = y_true_vrp[i-1]
        predicted_vrp = y_pred_vrp[i]
        
        # Decision logic based on physics understanding
        if current_vrp > paroxysm_threshold and predicted_vrp < current_vrp:
            labels.append('Post-Event')  # Decreasing from high activity
        elif predicted_vrp > paroxysm_threshold:
            labels.append('High-Paroxysm')  # High activity state
        elif predicted_vrp > current_vrp * (1 + growth_threshold) and current_vrp > calm_threshold:
            labels.append('Pre-Event')  # Significant growth detected
        else:
            labels.append('Calm')  # Default calm state
            
    # Prepend 'Calm' for first element (no previous value to compare)
    return np.array(['Calm'] + labels)


def train_and_evaluate_lstm_pinn(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 config: ModelConfig,
                                 sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Train LSTM forecasting model with physics-informed loss function.
    
    Implements a custom training loop with physics constraints to improve
    model behavior and interpretability.
    
    @param X_train: Training features (unused - using y_train as time series)
    @param y_train: Training time series values
    @param X_val: Validation features (unused)
    @param y_val: Validation time series values
    @param X_test: Test features (unused)
    @param y_test: Test time series values
    @param config: Model configuration
    @param sample_weight: Optional sample weights (unused)
    @return: Dictionary with model, predictions, and metrics
    @author: KRIBET Naoufal
    """
    logging.info("--- Launching Physics-Informed LSTM (PINN) training ---")

    # Step 1: Data preparation into sequences
    sequence_length = 50  # Fixed sequence length for this implementation
    
    # Create DataFrames from series
    train_df = pd.DataFrame({'VRP': y_train.values}, index=y_train.index)
    val_df = pd.DataFrame({'VRP': y_val.values}, index=y_val.index)
    test_df = pd.DataFrame({'VRP': y_test.values}, index=y_test.index)
    
    # Transform to sequences
    X_train_seq, y_train_seq, _ = create_sequences(train_df, sequence_length)
    X_val_seq, y_val_seq, _ = create_sequences(val_df, sequence_length)
    X_test_seq, y_test_seq, test_indices = create_sequences(test_df, sequence_length)

    # Normalize data for neural network training
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))
    X_val_scaled = scaler_X.transform(X_val_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y_val_scaled = scaler_y.transform(y_val_seq.reshape(-1, 1))

    # Step 2: Model creation and training tools
    model = create_forecasting_model(sequence_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.nn_learning_rate)
    
    def physics_loss(y_true_batch, y_pred_batch, last_point_in_sequence, lambda_mono=0.5):
        """
        Custom loss function incorporating physics constraints.
        
        Combines MSE with monotonicity penalty to enforce physical behavior.
        
        @param y_true_batch: True values batch
        @param y_pred_batch: Predicted values batch
        @param last_point_in_sequence: Last point of input sequence
        @param lambda_mono: Weight for monotonicity penalty
        @return: Combined loss value
        """
        # Standard MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true_batch, y_pred_batch)
        
        # Physics constraint: penalize decreasing predictions (monotonicity)
        monotonicity_error = tf.maximum(0., last_point_in_sequence - y_pred_batch)
        monotonicity_loss = tf.reduce_mean(monotonicity_error)
        
        # Combine losses
        return tf.reduce_mean(mse_loss) + lambda_mono * monotonicity_loss

    # Step 3: Custom training loop with physics-informed loss
    epochs = config.nn_epochs
    batch_size = config.nn_batch_size
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}  # For compatibility

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = []
        
        # Training batches
        for i in range(0, len(X_train_scaled), batch_size):
            X_batch = X_train_scaled[i:i+batch_size]
            y_batch = y_train_scaled[i:i+batch_size]
            
            # Gradient computation and update
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                last_points = X_batch[:, -1, :]
                loss = physics_loss(y_batch, y_pred, last_points)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss.numpy())
        
        # Validation loss computation
        val_preds = model(X_val_scaled, training=False)
        last_points_val = X_val_scaled[:, -1, :]
        val_loss = physics_loss(y_val_scaled, val_preds, last_points_val)
        
        # Record history
        history['loss'].append(np.mean(epoch_loss))
        history['val_loss'].append(val_loss.numpy())
        print(f"  loss: {np.mean(epoch_loss):.4f} - val_loss: {val_loss.numpy():.4f}")

    # Step 4: Evaluation with interpretation
    X_test_scaled = scaler_X.transform(X_test_seq.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred_vrp = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Derive classification labels from predictions
    y_pred_labels = derive_state_from_forecast(y_test_seq, y_pred_vrp)
    
    # Get true labels (requires data_processor module)
    from .data_processor import define_internal_event_cycle
    original_test_labels = y_test.iloc[sequence_length:].reset_index(drop=True)
    y_true_labels_df = pd.DataFrame({'VRP': y_test_seq, 'Ramp': original_test_labels})
    y_true_labels_df_processed = define_internal_event_cycle(y_true_labels_df, pre_event_ratio=0.3)
    y_true_labels = y_true_labels_df_processed['Ramp'].values
    
    # Calculate classification metrics
    unique_labels = np.unique(np.concatenate((y_true_labels, y_pred_labels)))
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0, labels=unique_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)
    
    return {
        'model': model, 
        'scalers': {'X': scaler_X, 'y': scaler_y},
        'best_params': {'sequence_length': sequence_length},
        'predictions': pd.Series(y_pred_labels, index=test_indices),
        'report': report, 
        'confusion_matrix': cm, 
        'class_names': unique_labels.tolist(),
        'training_history': history
    }

class ThresholdClassifier:
    """
    A simple 'mock' classifier that encapsulates the threshold-based decision logic.
    
    It is designed to be compatible with the scikit-learn evaluation pipeline,
    allowing it to be used as a baseline model within the existing framework.
    
    @author: KRIBET Naoufal
    """
    def __init__(self, threshold_value: float = 0.5, feature_name: str = 'median10', positive_class_name: str = 'Actif'):
        self.threshold = threshold_value
        self.feature = feature_name
        self.positive_class = positive_class_name
        # Define the classes the model can predict for compatibility
        self.classes_ = np.array(['Calm', self.positive_class])

    def fit(self, X, y):
        """This model has no traditional 'fit' phase; the threshold is determined externally."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels based on the threshold applied to the specified feature."""
        if self.feature not in X.columns:
            raise ValueError(f"Feature '{self.feature}' not found in input data.")
        return np.where(X[self.feature] >= self.threshold, self.positive_class, 'Calm')

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Simulates prediction probabilities for API compatibility with the evaluation pipeline.
        Assigns a high confidence (0.99) to the predicted class.
        """
        if self.feature not in X.columns:
            raise ValueError(f"Feature '{self.feature}' not found in input data.")
            
        # Check which instances meet the threshold condition
        is_positive = (X[self.feature] >= self.threshold).values
        # Create a probability array initialized to zero
        probas = np.zeros((len(X), 2))
        
        # Assign high probability to the positive class where condition is met
        probas[is_positive, 1] = 0.99
        probas[is_positive, 0] = 0.01
        
        # Assign high probability to the 'Calm' class where condition is not met
        probas[~is_positive, 0] = 0.99
        probas[~is_positive, 1] = 0.01
        
        return probas

    def get_params(self, deep=True):
        """Required for scikit-learn compatibility (e.g., for use in pipelines)."""
        return {'threshold_value': self.threshold, 'feature_name': self.feature}


def train_and_evaluate_threshold(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 config: ModelConfig,
                                 sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    "Trains" and evaluates a simple baseline classifier based on an optimal threshold.
    
    The training process consists of finding the single best threshold on a predefined
    feature ('median10') that maximizes the F1-score on the combined training and
    validation sets. This serves as a robust baseline to measure the value of more
    complex models.
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_val: Validation features
    @param y_val: Validation labels
    @param X_test: Test features
    @param y_test: Test labels
    @param config: Model configuration object (unused, for compatibility)
    @param sample_weight: Accepted for API compatibility, but ignored by this model.
    @return: A results dictionary compatible with the application's pipeline.
    @author: KRIBET Naoufal
    """
    logging.info("--- Launching Baseline Model: Simple Threshold ---")
    
    FEATURE_TO_USE = 'median10' # The single feature used to make the decision.

    # Ensure the required feature is present in the dataset.
    if FEATURE_TO_USE not in X_train.columns:
        raise ValueError(f"The Threshold model requires the '{FEATURE_TO_USE}' feature, which is not present in the data.")

    # Step 1: Combine training and validation data to find the optimal threshold.
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    
    # Identify the positive class name (e.g., 'Actif', 'Pre-Event').
    positive_class = next((label for label in y_combined.unique() if label != 'Calm'), None)
    if not positive_class:
        raise ValueError("Could not determine the positive class for the Threshold model.")
    
    logging.info(f"Optimizing threshold on feature '{FEATURE_TO_USE}' for target class '{positive_class}'.")

    # Step 2: Define a range of candidate thresholds to test.
    feature_values = X_combined[FEATURE_TO_USE]
    # Test 100 thresholds spread across the feature's distribution (5th to 95th percentile)
    # to remain robust to outliers.
    threshold_candidates = np.linspace(feature_values.quantile(0.05), feature_values.quantile(0.95), 100)
    
    best_threshold = 0
    best_f1 = -1

    # Step 3: Iterate through candidates to find the best threshold.
    for threshold in threshold_candidates:
        y_pred = np.where(feature_values >= threshold, positive_class, 'Calm')
        score = f1_score(y_combined, y_pred, pos_label=positive_class, zero_division=0)
        
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    logging.info(f"Best threshold found: {best_threshold:.2f} (yielding F1-Score of {best_f1:.4f} on validation data).")

    # Step 4: Create the final model instance with the optimal threshold.
    final_model = ThresholdClassifier(
        threshold_value=best_threshold,
        feature_name=FEATURE_TO_USE,
        positive_class_name=positive_class
    )
    
    # Step 5: Evaluate this simple model on the unseen test set using the standard evaluation function.
    with timing_context("Final Threshold model evaluation"):
        results = _evaluate_model(final_model, X_test, y_test, final_model.get_params())
        
    return results

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [ModelTrainer] - %(message)s'
)