# --- FICHIER : Core/explainer.py ---
# VERSION MISE À JOUR POUR L'EXPLICABILITÉ GLOBALE

import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    """
    Encapsule un modèle entraîné et les données nécessaires pour générer
    des explications SHAP locales (pour un point) et globales (pour le modèle).
    """
    # Dans Core/explainer.py, REMPLACEZ cette méthode

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
        
        # --- CORRECTION DE COMPATIBILITÉ ---
        try:
            # Essayer d'initialiser avec l'argument pour les versions récentes
            self.explainer = shap.TreeExplainer(self.model, self.background_data, check_additivity=False)
        except TypeError:
            # Si ça échoue, c'est une ancienne version. On initialise sans l'argument.
            print("AVERTISSEMENT : Version de SHAP ancienne détectée. L'erreur d'additivité pourrait réapparaître.")
            self.explainer = shap.TreeExplainer(self.model, self.background_data)
        # --- FIN DE LA CORRECTION ---
        
        print("Explainer SHAP prêt.")

        print("Pré-calcul des valeurs SHAP pour tout le jeu de test...")
        try:
            # On met le calcul dans un try/except au cas où l'erreur d'additivité
            # se produirait sur l'ancienne version.
            self.shap_values = self.explainer.shap_values(self.X_test_for_shap)
        except Exception as e:
            if "Additivity check failed" in str(e):
                print("\nERREUR CRITIQUE : L'erreur d'additivité de SHAP s'est produite.")
                print("Veuillez mettre à jour votre bibliothèque SHAP avec : pip install --upgrade shap")
                # On propage l'erreur pour que l'application ne continue pas avec un explainer cassé.
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
        # Pour les modèles multi-classes, self.shap_values est une liste de matrices.
        # On extrait la bonne ligne (correspondant à l'instance) de chaque matrice.
        shap_values_for_instance = [sv[instance_index] for sv in self.shap_values]
        
        # On récupère les données de la feature correspondante
        instance_data = self.X_test_for_shap.iloc[instance_index]
        
        # La valeur de base (expected_value) est la même pour toutes les prédictions
        base_value = self.explainer.expected_value
        
        return shap_values_for_instance, base_value, instance_data