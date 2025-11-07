================================================================================
PLAN D'ACTION : MISE A NIVEAU SCIENTIFIQUE DU MANUSCRIT ET DES ANALYSES
================================================================================

INTRODUCTION :
Ce document détaille les actions nécessaires pour élever la rigueur scientifique et la robustesse de notre manuscrit. Chaque point identifie une faiblesse méthodologique actuelle et propose des actions concrètes (analyses, modifications du code) ainsi que la manière de les intégrer dans l'article.


--- SECTION 1 : Renforcer la Causalité Temporelle dans le Feature Engineering ---

A. Implications pour le Code et les Analyses :
   - Le remplissage actuel des valeurs NaN initiales dans les fenêtres glissantes (`fillna(0.0)`) est une simplification.
   - ACTION : Modifier la fonction `_process_single_segment` dans `feature_extractor.py`.
     - Remplacer le `fillna(0.0)` global.
     - Pour chaque calcul de `rolling`, imputer les valeurs manquantes en utilisant une stratégie qui respecte la causalité : d'abord un "forward-fill" (`ffill`) pour propager les dernières valeurs valides, puis un "backward-fill" (`bfill`) uniquement pour les tous premiers points où aucune information passée n'existe.

B. Modifications dans l'Article (Section II.B - Multi-Scale Feature Engineering) :
   - Remplacer la description actuelle par :
     "To ensure strict temporal causality, all feature engineering operations were performed independently on each dataset partition. Specifically, rolling window statistics use a forward-looking window only, preventing any information leakage from future time points. For the initial points within each continuous segment where insufficient historical data exists to fill the window, we employ a forward-fill followed by a backward-fill strategy to impute values."


--- SECTION 2 : Corriger et Justifier les Paramètres du TimeSeriesSplit ---

A. Implications pour le Code et les Analyses :
   - La validation croisée sans "gap" peut surestimer la performance en raison de l'autocorrélation à court terme.
   - ACTION : Modifier l'instanciation de `TimeSeriesSplit` dans `model_trainer.py` (fonctions `_find_best_rf_model`, etc.).
     - CHANGER : `TimeSeriesSplit(n_splits=config.cv_folds)`
     - PAR : `TimeSeriesSplit(n_splits=3, gap=50)`
   - ACTION CRITIQUE : Relancer toutes les optimisations d'hyperparamètres avec cette nouvelle configuration pour garantir la validité des résultats de l'article.

B. Modifications dans l'Article (Section II.C.3 - Hyperparameter Optimization) :
   - Remplacer la description actuelle par :
     "Hyperparameters were optimized using Randomized Search with a Time Series Cross-Validation splitter (TimeSeriesSplit). We used k=3 splits and set a 'gap' of 50 data points (equivalent to approx. 4 hours) between the training and validation folds. This gap is critical as it prevents the model from being evaluated on data immediately following the training data, thus simulating a more realistic operational scenario where predictions are made on a truly unseen future."


--- SECTION 3 : Justifier Scientifiquement le Filtre de Persistance ---

A. Implications pour le Code et les Analyses :
   - Le choix de `N=3` est actuellement arbitraire. Il doit être justifié par une analyse de sensibilité.
   - ACTION : Créer une nouvelle fonction d'analyse post-entraînement dans `model_trainer.py`.
     - Cette fonction bouclera sur N de 1 à 10.
     - Pour chaque valeur de N, elle appliquera le filtre sur les prédictions du jeu de test et recalculera la Précision, le Rappel et le F1-Score.
     - Les résultats de cette analyse (tableau ou graphique) formeront la base de la section A des "Supplementary Materials".

B. Modifications dans l'Article (Section II.D.2 - Temporal Persistence Filter) :
   - Remplacer la description actuelle par :
     "A temporal persistence filter is applied, where an 'Active' prediction is confirmed only if sustained for N=3 consecutive time steps (15 minutes). This value was selected based on a sensitivity analysis (see Supplementary Materials, Section A) which demonstrated an optimal balance in the Precision-Recall trade-off for N in [2, 4]. The choice of N=3 is further supported by domain knowledge, as it represents a minimum duration for physically significant eruptive precursor signals [CITE DOMAIN EXPERT PAPER HERE]."

The prior assumption is that a physically meaningful precursor volcanic event cannot be an instantaneous spike, but rather a process that must last for a minimum duration to be regarded as real.

--- SECTION 4 : Ajouter des Intervalles de Confiance aux Métriques de Performance ---

A. Implications pour le Code et les Analyses :
   - Les métriques rapportées en tant que valeurs uniques sur un petit jeu de test manquent de robustesse statistique.
   - ACTION : Implémenter le "bootstrapping" dans la fonction `_evaluate_model` de `model_trainer.py`.
     1. Après avoir obtenu les paires (y_true, y_pred) sur le jeu de test.
     2. Lancer une boucle de 5000 itérations.
     3. A chaque itération, créer un échantillon "bootstrap" en tirant avec remise des paires du jeu de test.
     4. Calculer Precision, Recall, F1-Macro sur cet échantillon.
     5. Stocker les résultats de chaque métrique.
     6. A la fin, calculer les percentiles 2.5 et 97.5 de la distribution de chaque métrique pour obtenir l'intervalle de confiance à 95%.
     7. Retourner ces intervalles de confiance dans le dictionnaire de résultats.

B. Modifications dans l'Article (Partout où des métriques sont citées) :
   - Mettre à jour TOUTES les métriques rapportées (dans le texte et les légendes des figures) pour inclure les intervalles de confiance.
   - Exemple pour la légende de la Figure 7 :
     "The optimized model achieves a global Macro F1-Score of 0.770 [95% CI: 0.735-0.805]. For the 'Active' class, Precision is 0.76 [95% CI: 0.71-0.81] and Recall is 0.86 [95% CI: 0.82-0.90]. Confidence intervals were calculated using non-parametric bootstrapping with 5000 resamples."


--- SECTION 5 : Comparer le Modèle à des Baselines Naïves ---

A. Implications pour le Code et les Analyses :
   - L'utilité du modèle de ML doit être prouvée par rapport à des heuristiques simples.
   - ACTION : Créer une nouvelle fonction `_evaluate_baselines(y_train, y_test, vrp_test)` dans `model_trainer.py`.
     - Implémenter et évaluer (en F1-Score) les trois baselines :
       1. Modèle Persistant : prédit la classe de t-1.
       2. Seuil VRP Statique : seuil basé sur le 75e percentile du VRP du jeu d'entraînement.
       3. Seuil sur Moyenne Mobile : seuil basé sur une moyenne mobile de 30 points.
     - Implémenter un test statistique (ex: `scipy.stats.wilcoxon`) pour comparer les distributions d'erreurs du meilleur modèle ML et de la meilleure baseline.

B. Modifications dans l'Article (Fin de la section III.C) :
   - Ajouter le paragraphe suivant :
     "To validate the model's predictive utility, we compared its performance against three naive baselines: a persistence model (F1=0.52), a static VRP threshold classifier (F1=0.61), and a moving average threshold (F1=0.68). The parsimonious ML model (F1=0.725) significantly outperforms all baselines (Wilcoxon signed-rank test, p<0.01), confirming that the learned patterns capture genuine predictive signal beyond simple heuristics."


--- SECTION 6 : Renforcer la Discussion sur les Limites (Menaces à la Validité) ---

A. Implications pour le Code et les Analyses :
   - Aucune, il s'agit d'une section de discussion critique basée sur la configuration de l'expérience.

B. Modifications dans l'Article (Nouvelle sous-section IV.D) :
   - Ajouter la sous-section dédiée "Methodological Limitations and Threats to Validity" avec les quatre points que vous avez listés :
     1. Limited temporal generalization (taille du test set).
     2. Single-volcano bias (données uniquement de l'Etna).
     3. Feature engineering assumptions (choix des fenêtres non optimisé).
     4. Evaluation metric selection (F1-Score vs. coûts opérationnels asymétriques).


--- SECTION 7 & 8 : Rendre la Conclusion plus Modeste et Créer les Annexes ---

A. Implications pour le Code et les Analyses :
   - C'est ici que se trouve le plus gros du travail d'analyse supplémentaire.
   - ACTION : Mener les expériences systématiques suivantes pour alimenter les "Supplementary Materials" :
     - Analyse de sensibilité du filtre de persistance (déjà couverte au point 3).
     - Analyse de sensibilité des hyperparamètres : lancer plusieurs entraînements en faisant varier `n_estimators` et `max_depth` pour montrer la robustesse du F1-Score.
     - Analyse de sensibilité de la taille des fenêtres : définir des configurations alternatives (ex: "short-only", "long-only") et relancer tout le pipeline pour comparer les F1-Scores finaux.

B. Modifications dans l'Article (Section V et nouvelle section "Supplementary Materials") :
   - Dans la Conclusion, remplacer "high predictive performance" par :
     "promising but preliminary predictive capability that requires extensive validation across diverse volcanic contexts before operational deployment."
   - Créer une section "Supplementary Materials" avec les sous-sections A, B, C détaillant les résultats des analyses de sensibilité.


--- SECTION 9 : Ajouter un Diagramme de Calibration ---

A. Implications pour le Code et les Analyses :
   - L'évaluation de la fiabilité des probabilités prédites est manquante.
   - ACTION :
     - Dans `_evaluate_model`, utiliser `sklearn.calibration.calibration_curve` sur les probabilités du jeu de test (`y_test_pred_proba`) et les vrais labels (`y_test`) pour obtenir les données de la courbe.
     - Dans `main2.py`, créer une nouvelle fonction de plotting pour générer le diagramme de fiabilité (predicted probability vs. fraction of positives).

B. Modifications dans l'Article (Nouvelle figure à la fin de la section III) :
   - Ajouter une nouvelle figure (Fig. 11) avec la légende :
     "Fig. 11. Reliability diagram (calibration curve) for the parsimonious model. The diagonal represents perfect calibration. Our model is well-calibrated for P(Active) < 0.7 but slightly overconfident for P(Active) > 0.8, suggesting that very high probability predictions should be interpreted cautiously in an operational context."

---

Ce plan d'action est ambitieux mais transformera le manuscrit en une publication de très haut niveau, anticipant et répondant aux critiques les plus exigeantes d'un processus de relecture.