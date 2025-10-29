import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    """
    Encapsule un modèle entraîné et les données nécessaires pour générer
    des explications SHAP locales (pour un point) et globales (pour le modèle).
    """

    def __init__(self, model, background_data: pd.DataFrame, X_test_for_shap: pd.DataFrame):
        """
        Initialise l'explainer et pré-calcule les valeurs SHAP.
        VERSION CORRIGÉE : Compatible avec les anciennes et nouvelles versions de SHAP.
        """
        if not hasattr(model, 'predict_proba'):
            raise TypeError("Le modèle doit avoir une méthode 'predict_proba'.")
            
        self.model = model
        self.background_data = background_data
        self.X_test_for_shap = X_test_for_shap
        
        print("Initialisation de l'explainer SHAP...")

        try:
            self.explainer = shap.TreeExplainer(self.model, self.background_data, check_additivity=False)
        except TypeError:
            print("AVERTISSEMENT : Version de SHAP ancienne détectée. L'erreur d'additivité pourrait réapparaître.")
            self.explainer = shap.TreeExplainer(self.model, self.background_data)
        
        print("Explainer SHAP prêt.")

        print("Pré-calcul des valeurs SHAP pour tout le jeu de test...")
        try:
            self.shap_values = self.explainer.shap_values(self.X_test_for_shap)
        except Exception as e:
            if "Additivity check failed" in str(e):
                print("\nERREUR CRITIQUE : L'erreur d'additivité de SHAP s'est produite.")
                print("Veuillez mettre à jour votre bibliothèque SHAP avec : pip install --upgrade shap")
                raise e
            else:
                raise e
        print("Pré-calcul des valeurs SHAP terminé.")
    def explain_instance_by_index(self, instance_index: int) -> tuple:
        """
        Récupère les valeurs SHAP pré-calculées pour une seule instance via son index.

        Args:
            instance_index (int): L'index (position dans la table) de la prédiction à expliquer.

        Returns:
            Un tuple contenant (shap_values_for_instance, base_value, instance_data).
        """
        shap_values_for_instance = [sv[instance_index] for sv in self.shap_values]
        
        instance_data = self.X_test_for_shap.iloc[instance_index]
        
        base_value = self.explainer.expected_value
        

        return shap_values_for_instance, base_value, instance_data
