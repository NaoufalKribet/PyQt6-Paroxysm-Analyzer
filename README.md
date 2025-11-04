================================================================================
PyQt6-Paroxysm-Analyzer: Predictive Analysis of Paroxysmal Events
================================================================================

PROJECT OVERVIEW
----------------

This project is a comprehensive desktop application developed with PyQt6 for the analysis of Volcanic Radiated Power (VRP) time series. It provides a full end-to-end Machine Learning pipeline, from data loading and preparation to model training, evaluation, and, most importantly, interpretation.

The primary goal is to provide domain experts (e.g., volcanologists, geophysicists) with a scientifically rigorous and robust tool to test hypotheses, explore potential precursors to paroxysmal events, and understand the decisions made by complex predictive models. The application is designed with a strong emphasis on methodological best practices, such as preventing data leakage and ensuring model explainability.


CORE FEATURES
-------------

- Data Loading & Visualization: Import data from Excel files and instantly visualize time series and descriptive statistics.

- Interactive Feature Engineering: Generate dozens of temporal features (statistical, dynamic, expert-derived) over configurable rolling windows. Analyze their distributions and correlations.

- Multi-Model Training: Train and evaluate a variety of models, including Random Forest, LightGBM, K-Nearest Neighbors (KNN), and Neural Networks (via TensorFlow/Keras).

- Rigorous Temporal Validation: Strictly uses TimeSeriesSplit cross-validation to prevent temporal data leakage, ensuring an honest and reliable estimation of model performance on unseen data.

- Integrated Explainability (XAI):
  - Feature Importance: Compare Gini Importance (MDI) with the more robust Permutation Feature Importance (PFI).
  - Local Explanations (SHAP): For any single prediction, visualize exactly which features contributed and by how much using SHAP waterfall plots.

- Real-Time Simulation: Generate realistic synthetic data and watch a trained model make predictions in a simulated real-time environment.

- 'What-If' Scenarios: Interactively edit a time series to create a custom scenario and analyze the immediate impact of these changes on the model's predictive probabilities.

- Asynchronous & Responsive UI: All long-running tasks (data loading, feature extraction, model training) are executed in background threads. This ensures the user interface never freezes and remains fully responsive, with progress and cancellation dialogs for every task.


TECHNICAL STACK
---------------

- Language: Python 3.9+
- GUI Framework: PyQt6
- Data Science Stack:
  - Pandas & NumPy for data manipulation.
  - Scikit-learn for core Machine Learning pipelines.
  - TensorFlow (Keras) for Neural Network models.
- Visualization: Plotly for high-quality, interactive charts.
- Model Interpretability: SHAP (SHapley Additive exPlanations).


INSTALLATION AND USAGE
----------------------

This project uses standard Python tools and can be run on Windows, macOS, or Linux.

1. Prerequisites:
   - Ensure you have Python 3.9 or a later version installed.
   - It is highly recommended to use a virtual environment.

   # Create and activate a virtual environment
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate

2. Clone the Repository:
   git clone https://github.com/your-username/PyQt6-Paroxysm-Analyzer.git
   cd PyQt6-Paroxysm-Analyzer

3. Install Dependencies:
   The `requirements.txt` file lists all necessary libraries.

   pip install -r requirements.txt

   Note: The installation of TensorFlow and PyQt6 may take a few minutes.

4. Launch the Application:
   The main entry point is `run_app.py`.

   python run_app.py

   This will open the application's launch menu.


PROJECT ARCHITECTURE
--------------------

The project is structured following the Model-View-Controller (MVC) architectural pattern to ensure a clear separation of concerns, making the code more maintainable and scalable.

File Structure:

PyQt6-Paroxysm-Analyzer/
|
+-- Core/                   # The "Model" (business logic, algorithms)
|   +-- data_processor.py
|   +-- feature_extractor.py
|   +-- model_trainer.py
|   +-- model_manager.py
|   +-- explainer.py
|   +-- ...
|
+-- models/                 # Default directory for saved model artifacts
|
+-- app_controller.py       # The "Controller" (orchestrates interactions)
+-- main2.py                # The "View" (all GUI and visualization code)
+-- run_app.py              # Application entry point
+-- launch_menu.py          # Initial selection menu
+-- ui_dialogs.py           # Custom dialogs (e.g., progress bars)
+-- requirements.txt        # Python dependencies


- Core/: This is the scientific heart of the project. This module is entirely independent of the user interface and could be reused in other contexts (e.g., a batch script, a web API). It contains all the logic for data processing, feature engineering, model training, and explanation.

- app_controller.py: This is the "Controller". It acts as the bridge between the user's actions in the View and the computations performed by the Model. It manages the application's state and the lifecycle of asynchronous tasks.

- main2.py: This is the "View". It contains all the code for building the graphical user interface using PyQt6 and displaying interactive plots with Plotly. It contains no business logic.

- run_app.py: The simple script that initializes the application objects and starts the event loop.


AUTHOR
------
Naoufal Kribet


LICENSE
-------
This project is licensed under the MIT License. See the LICENSE file for full details.