"""
Model Explainability Module using SHAP
=======================================

This module provides interpretability tools for machine learning models using
SHAP (SHapley Additive exPlanations) values. 

UPDATED VERSION 3.0: Now supports multiple SHAP explainer types to handle
both tree-based models (RandomForest, LightGBM) and other model types like
K-Nearest Neighbors (KNN), which are not supported by TreeExplainer.

@author: KRIBET Naoufal
@affiliation: 5th year Engineering Student, EOST (École et Observatoire des Sciences de la Terre)
@date: 2025-11-17
@version: 3.0
"""

import shap
import pandas as pd
import numpy as np
import logging

class ModelExplainer:
    """
    Encapsulates a trained model and provides SHAP-based explanations.
    
    This class wraps a trained ML model, detects its type, and initializes the
    appropriate SHAP explainer (TreeExplainer for tree models, KernelExplainer for others).
    It pre-computes SHAP values for efficient generation of local explanations.
    """
    
    def __init__(self, model, background_data: pd.DataFrame, X_test_for_shap: pd.DataFrame):
        """
        Initializes the correct explainer based on model type and pre-computes SHAP values.
        
        @param model: Trained model (e.g., RandomForest, XGBoost, KNeighborsClassifier)
        @param background_data: Representative sample of data for SHAP baseline.
        @param X_test_for_shap: Test dataset for which SHAP values will be pre-computed.
        @raises TypeError: If the model does not have a 'predict_proba' method.
        @raises Exception: If SHAP value computation fails.
        """
        if not hasattr(model, 'predict_proba'):
            raise TypeError("Model must have a 'predict_proba' method.")
            
        self.model = model
        self.background_data = background_data
        self.X_test_for_shap = X_test_for_shap
        
        # --- MODEL-AWARE EXPLAINER SELECTION ---
        model_type = type(self.model).__name__
        is_tree_model = hasattr(self.model, 'feature_importances_')

        logging.info(f"Initializing SHAP explainer for model type: {model_type}")

        if is_tree_model:
            logging.info("Tree-based model detected. Using shap.TreeExplainer.")
            try:
                # Modern SHAP with additivity check disabled
                self.explainer = shap.TreeExplainer(self.model, self.background_data, check_additivity=False)
            except TypeError:
                # Fallback for older SHAP versions
                logging.warning("Legacy SHAP version detected. Using TreeExplainer without additivity check parameter.")
                self.explainer = shap.TreeExplainer(self.model, self.background_data)
        else:
            logging.info(f"Non-tree model detected ({model_type}). Using shap.KernelExplainer. This may be slower.")
            # KernelExplainer requires a summary of the background data.
            # Using shap.kmeans is recommended for performance.
            background_summary = shap.kmeans(self.background_data, 10) # Summarize with 10 clusters
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background_summary)
        
        logging.info("SHAP explainer ready.")
        
        # --- PRE-COMPUTATION OF SHAP VALUES ---
        logging.info("Pre-computing SHAP values for the entire test set...")
        try:
            # The way shap_values are computed is consistent across explainers
            self.shap_values = self.explainer.shap_values(self.X_test_for_shap)
        except Exception as e:
            # Specific handling for additivity check failure remains useful for TreeExplainer
            if "Additivity check failed" in str(e):
                logging.critical("\nCRITICAL ERROR: SHAP additivity check failed.")
                logging.critical("Please update your SHAP library with: pip install --upgrade shap")
                raise e
            else:
                raise e
        logging.info("SHAP values pre-computation completed.")
    
    def explain_instance_by_index(self, instance_index: int) -> tuple:
        """
        Retrieves pre-computed SHAP values for a single instance by index.
        
        @param instance_index: Index (position in table) of the prediction to explain.
        @return: Tuple containing (shap_values_for_instance, base_value, instance_data).
        @note: Index refers to position in the X_test_for_shap DataFrame.
        """
        # Extract SHAP values for this specific instance across all classes.
        # This works for both Tree and Kernel explainers' output format.
        shap_values_for_instance = [sv[instance_index] for sv in self.shap_values]
        
        # Get original feature values for this instance.
        instance_data = self.X_test_for_shap.iloc[instance_index]
        
        # Get baseline prediction value. For KernelExplainer it's expected_value[0] per class.
        base_value = self.explainer.expected_value
        
        return shap_values_for_instance, base_value, instance_data