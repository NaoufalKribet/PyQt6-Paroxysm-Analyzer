"""
Model Explainability Module using SHAP
=======================================

This module provides interpretability tools for machine learning models using
SHAP (SHapley Additive exPlanations) values. It enables both local (instance-level)
and global (model-level) explanations for volcanic activity predictions.

@author: KRIBET Naoufal
@affiliation: 5th year Engineering Student, EOST (École et Observatoire des Sciences de la Terre)
@date: 2025-11-12
@version: 1.0
"""

import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    """
    Encapsulates a trained model and provides SHAP-based explanations.
    
    This class wraps a trained machine learning model and pre-computes SHAP values
    for efficient generation of both local explanations (single prediction) and
    global explanations (overall model behavior).
    
    The implementation is compatible with both legacy and modern versions of the
    SHAP library, handling known additivity check issues gracefully.
    """
    
    def __init__(self, model, background_data: pd.DataFrame, X_test_for_shap: pd.DataFrame):
        """
        Initializes the explainer and pre-computes SHAP values.
        
        CORRECTED VERSION: Compatible with both old and new SHAP library versions.
        The method handles the additivity check parameter gracefully for backward
        compatibility with older SHAP installations.
        
        @param model: Trained model with predict_proba method (e.g., RandomForest, XGBoost)
        @param background_data: Representative sample of training data for SHAP baseline
        @param X_test_for_shap: Test dataset for which SHAP values will be pre-computed
        @raises TypeError: If model does not have predict_proba method
        @raises Exception: If SHAP additivity check fails (requires SHAP library update)
        
        @note: SHAP values are pre-computed during initialization for efficiency
        """
        if not hasattr(model, 'predict_proba'):
            raise TypeError("Model must have a 'predict_proba' method.")
            
        self.model = model
        self.background_data = background_data
        self.X_test_for_shap = X_test_for_shap
        
        print("Initializing SHAP explainer...")
        try:
            # Attempt to initialize with additivity check disabled (modern SHAP)
            self.explainer = shap.TreeExplainer(self.model, self.background_data, check_additivity=False)
        except TypeError:
            # Fallback for older SHAP versions that don't support check_additivity parameter
            print("WARNING: Legacy SHAP version detected. Additivity error may reoccur.")
            self.explainer = shap.TreeExplainer(self.model, self.background_data)
        
        print("SHAP explainer ready.")
        print("Pre-computing SHAP values for entire test set...")
        try:
            self.shap_values = self.explainer.shap_values(self.X_test_for_shap)
        except Exception as e:
            if "Additivity check failed" in str(e):
                print("\nCRITICAL ERROR: SHAP additivity check failed.")
                print("Please update your SHAP library with: pip install --upgrade shap")
                raise e
            else:
                raise e
        print("SHAP values pre-computation completed.")
    
    def explain_instance_by_index(self, instance_index: int) -> tuple:
        """
        Retrieves pre-computed SHAP values for a single instance by index.
        
        This method provides local interpretability by extracting SHAP values
        for a specific prediction. SHAP values indicate how much each feature
        contributed to pushing the prediction away from the baseline.
        
        @param instance_index: Index (position in table) of the prediction to explain
        @return: Tuple containing (shap_values_for_instance, base_value, instance_data)
                 - shap_values_for_instance: SHAP values for each feature and class
                 - base_value: Expected model output (baseline prediction)
                 - instance_data: Original feature values for this instance
        
        @note: Index refers to position in X_test_for_shap DataFrame
        """
        # Extract SHAP values for this specific instance across all classes
        shap_values_for_instance = [sv[instance_index] for sv in self.shap_values]
        
        # Get original feature values for this instance
        instance_data = self.X_test_for_shap.iloc[instance_index]
        
        # Get baseline prediction value
        base_value = self.explainer.expected_value
        
        return shap_values_for_instance, base_value, instance_data