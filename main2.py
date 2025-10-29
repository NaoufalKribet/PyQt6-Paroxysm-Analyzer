# ###############################################################################################################################################
#
#   PROJET: Analyse Prédictive de l'Activité Volcanique à partir de donnée paraxysmique
#   FICHIER: main2.py
#   RÔLE: Point d'entrée principal et interface graphique (Vue MVC) de l'application.
#
#   AUTEUR: Naoufal Kribet
#   DATE: 26/08/2025
#   VERSION: 1.0
#
# ------------------------------------------------------------------------------------------------------------------------------------------------
#
#   DESCRIPTION:
#   Cette application est un outil d'aide à la décision pour l'analyse et la
#   prévision de l'activité volcanique du Mont Etna. Basée sur une approche
#   de Machine Learning (Random Forest) et un feature engineering robuste,
#   elle vise à fournir une analyse probabiliste des états du volcan pour
#   assister les experts.
#
#   STACK TECHNIQUE:
#   - Langage: Python 3
#   - Interface: PyQt6
#   - Data Science: Pandas, Scikit-learn, NumPy
#   - Visualisation: Plotly
#   - Explicabilité: SHAP, TensorFlow (pour Keras)
#
# #################################################################################################################################################


import sys
import pandas as pd
import numpy as np
import json
import os 
import tempfile
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QAbstractTableModel, Qt, pyqtSlot
from PyQt6.QtWidgets import (

    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableView, QFileDialog, QLabel, QTabWidget,
    QGroupBox, QFormLayout, QCheckBox, QSpinBox, QTextEdit,
    QLineEdit, QComboBox, QMessageBox, QSplitter, QListWidget,
    QTableWidget, QTableWidgetItem, QStackedWidget, QDoubleSpinBox, QRadioButton, QSlider, QMenu
)
from PyQt6.QtCore import QObject, pyqtProperty
from PyQt6.QtWebChannel import QWebChannel
import datetime
import shap 
from app_controller import AppController
from ui_dialogs import ProcessingDialog
import tensorflow
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Au début de main2.py

# --- SUPPRIMEZ IMPÉRATIVEMENT CES LIGNES S'IL EN RESTE ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from ui_dialogs import ProcessingDialog, ImageLoadingDialog
# --- FIN DE LA SUPPRESSION ---

# main2.py (au début du fichier)

import plotly.graph_objects as go
import plotly.io as pio

# ==============================================================================
# == DÉFINITION DU THÈME GRAPHIQUE POUR PUBLICATION SCIENTIFIQUE              ==
# ==============================================================================
publication_template = go.layout.Template()

publication_template.layout = {
    'font': {
        'family': "Times New Roman, serif", # Police classique pour les articles
        'size': 14,                         # Taille de police lisible
        'color': "black"
    },
    'paper_bgcolor': 'white',               # Fond de la figure
    'plot_bgcolor': 'white',                # Fond du graphique
    'title': {
        'font': {'size': 18},
        'x': 0.5                            # Titre centré
    },
    'xaxis': {
        'showgrid': True, 'gridcolor': '#e5e5e5', 'gridwidth': 0.5,
        'linecolor': 'black', 'zerolinecolor': '#888', 'zerolinewidth': 0.7,
        'ticks': 'outside', 'mirror': True
    },
    'yaxis': {
        'showgrid': True, 'gridcolor': '#e5e5e5', 'gridwidth': 0.5,
        'linecolor': 'black', 'zerolinecolor': '#888', 'zerolinewidth': 0.7,
        'ticks': 'outside', 'mirror': True
    },
    'legend': {
        'bgcolor': 'rgba(255,255,255,0.7)',
        'bordercolor': 'black',
        'borderwidth': 0.5
    },
    # Palette de couleurs contrastées et compatibles noir et blanc
    'colorway': ['#003f5c', '#ff6361', '#58508d', '#ffa600', '#bc5090', '#7a5195', '#ef5675', '#28519e']
}

# Enregistrer le template pour pouvoir l'utiliser facilement
pio.templates['publication'] = publication_template
# ==============================================================================

# ... (le reste de vos imports et de votre code)
DARK_STYLESHEET = """
    /* ================================
       PARAMÈTRES GLOBAUX
       ================================ */
    QWidget {
        background-color: #1a1a1a;
        color: #e8e8e8;
        font-family: 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, -apple-system, sans-serif;
        font-size: 10pt;
        font-weight: 400;
        selection-background-color: #4a4a4a;
        selection-color: #ffffff;
    }

    /* Fenêtre principale */
    QMainWindow {
        background-color: #0f0f0f;
        border: 1px solid #2a2a2a;
    }

    /* ================================
       GROUPBOX - Style moderne
       ================================ */
    QGroupBox {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #2a2a2a, stop: 1 #1f1f1f);
        border: 2px solid #404040;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 8px;
        font-weight: 600;
        font-size: 11pt;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 12px;
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                   stop: 0 #404040, stop: 0.5 #505050, stop: 1 #404040);
        border: 1px solid #606060;
        border-radius: 6px;
        color: #f5f5f5;
        margin-left: 8px;
        font-weight: 700;
    }

    QGroupBox:disabled {
        color: #666666;
        border-color: #2a2a2a;
    }

    /* ================================
       ONGLETS - Design sophistiqué
       ================================ */
    QTabWidget::pane {
        border: 2px solid #404040;
        border-radius: 6px;
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #2a2a2a, stop: 1 #1a1a1a);
        margin-top: -1px;
    }

    QTabBar::tab {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #3a3a3a, stop: 0.5 #2f2f2f, stop: 1 #2a2a2a);
        border: 1px solid #505050;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        min-width: 12ex;
        padding: 8px 16px;
        margin-right: 2px;
        font-weight: 500;
        color: #cccccc;
    }

    QTabBar::tab:selected {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #505050, stop: 0.5 #454545, stop: 1 #404040);
        border-color: #707070;
        color: #ffffff;
        font-weight: 600;
        margin-bottom: -2px;
    }

    QTabBar::tab:hover:!selected {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #454545, stop: 0.5 #3a3a3a, stop: 1 #353535);
        color: #e0e0e0;
    }

    /* ================================
       BOUTONS - Style premium
       ================================ */
    QPushButton {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #4a4a4a, stop: 0.5 #404040, stop: 1 #353535);
        color: #ffffff;
        border: 1px solid #606060;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 10pt;
        min-height: 20px;
    }

    QPushButton:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #555555, stop: 0.5 #4a4a4a, stop: 1 #404040);
        border-color: #707070;
        color: #ffffff;
    }

    QPushButton:pressed {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #2a2a2a, stop: 0.5 #303030, stop: 1 #353535);
        border-color: #808080;
        padding-top: 9px;
        padding-bottom: 7px;
    }

    QPushButton:disabled {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #2a2a2a, stop: 1 #252525);
        color: #666666;
        border-color: #3a3a3a;
    }

    QPushButton:focus {
        border: 2px solid #707070;
        outline: none;
    }

    /* Boutons spéciaux */
    QPushButton[class="primary"] {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #5a5a5a, stop: 0.5 #4f4f4f, stop: 1 #454545);
        font-weight: 700;
    }

    QPushButton[class="danger"] {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #704040, stop: 0.5 #603535, stop: 1 #502a2a);
        border-color: #805050;
    }

    /* ================================
       CHAMPS DE SAISIE - Design raffiné
       ================================ */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #1f1f1f, stop: 1 #2a2a2a);
        border: 2px solid #404040;
        border-radius: 5px;
        padding: 6px 10px;
        color: #e8e8e8;
        font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
        selection-background-color: #555555;
    }

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #707070;
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #252525, stop: 1 #2f2f2f);
    }

    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
        background-color: #1a1a1a;
        color: #666666;
        border-color: #2a2a2a;
    }

    /* SpinBox et DoubleSpinBox */
    QSpinBox, QDoubleSpinBox {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #1f1f1f, stop: 1 #2a2a2a);
        border: 2px solid #404040;
        border-radius: 5px;
        padding: 4px 8px;
        color: #e8e8e8;
        font-family: 'JetBrains Mono', monospace;
    }

    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #555555;
        border-bottom: 1px solid #555555;
        border-top-right-radius: 3px;
        background: #404040;
    }

    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid #555555;
        border-top: 1px solid #555555;
        border-bottom-right-radius: 3px;
        background: #404040;
    }

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 4px solid #cccccc;
    }

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #cccccc;
    }

    /* ComboBox */
    QComboBox {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #2a2a2a, stop: 1 #1f1f1f);
        border: 2px solid #404040;
        border-radius: 5px;
        padding: 6px 10px;
        color: #e8e8e8;
        min-width: 6em;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 25px;
        border-left: 1px solid #555555;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
        background: #404040;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #cccccc;
    }

    QComboBox QAbstractItemView {
        background-color: #2a2a2a;
        border: 2px solid #555555;
        selection-background-color: #505050;
        color: #e8e8e8;
    }

    /* ================================
       TABLES ET LISTES - Design moderne
       ================================ */
    QTableView, QTableWidget, QTreeView, QListWidget {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #2a2a2a, stop: 1 #1f1f1f);
        border: 2px solid #404040;
        border-radius: 6px;
        gridline-color: #353535;
        alternate-background-color: #252525;
        color: #e8e8e8;
        font-size: 9pt;
    }

    QTableView::item, QTableWidget::item, QTreeView::item, QListWidget::item {
        padding: 4px 8px;
        border-bottom: 1px solid #353535;
    }

    QTableView::item:selected, QTableWidget::item:selected, 
    QTreeView::item:selected, QListWidget::item:selected {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #505050, stop: 1 #454545);
        color: #ffffff;
    }

    QTableView::item:hover, QTableWidget::item:hover,
    QTreeView::item:hover, QListWidget::item:hover {
        background-color: #353535;
    }

    /* Headers */
    QHeaderView::section {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #404040, stop: 1 #353535);
        color: #f0f0f0;
        padding: 6px 12px;
        border: 1px solid #555555;
        border-radius: 0;
        font-weight: 600;
        font-size: 10pt;
    }

    QHeaderView::section:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #4a4a4a, stop: 1 #3f3f3f);
    }

    /* ================================
       BARRES DE DÉFILEMENT - Style premium
       ================================ */
    QScrollBar:vertical {
        border: none;
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                   stop: 0 #1a1a1a, stop: 1 #2a2a2a);
        width: 14px;
        margin: 16px 0 16px 0;
        border-radius: 7px;
    }

    QScrollBar::handle:vertical {
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                   stop: 0 #505050, stop: 0.5 #606060, stop: 1 #505050);
        min-height: 30px;
        border-radius: 6px;
        border: 1px solid #404040;
        margin: 1px;
    }

    QScrollBar::handle:vertical:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                   stop: 0 #606060, stop: 0.5 #707070, stop: 1 #606060);
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: #404040;
        height: 15px;
        border-radius: 7px;
        subcontrol-origin: margin;
    }

    QScrollBar::add-line:vertical:hover, QScrollBar::sub-line:vertical:hover {
        background: #505050;
    }

    QScrollBar:horizontal {
        border: none;
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #1a1a1a, stop: 1 #2a2a2a);
        height: 14px;
        margin: 0 16px 0 16px;
        border-radius: 7px;
    }

    QScrollBar::handle:horizontal {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #505050, stop: 0.5 #606060, stop: 1 #505050);
        min-width: 30px;
        border-radius: 6px;
        border: 1px solid #404040;
        margin: 1px;
    }

    /* ================================
       SÉPARATEURS ET SPLITTERS
       ================================ */
    QSplitter::handle {
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                   stop: 0 #404040, stop: 0.5 #505050, stop: 1 #404040);
        border: 1px solid #606060;
        border-radius: 2px;
    }

    QSplitter::handle:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                   stop: 0 #555555, stop: 0.5 #656565, stop: 1 #555555);
    }

    QSplitter::handle:vertical {
        height: 4px;
        margin: 2px 8px;
    }

    QSplitter::handle:horizontal {
        width: 4px;
        margin: 8px 2px;
    }

    /* ================================
       MENUS ET BARRES D'OUTILS
       ================================ */
    QMenuBar {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #3a3a3a, stop: 1 #2a2a2a);
        border-bottom: 1px solid #555555;
        color: #e8e8e8;
        padding: 2px;
    }

    QMenuBar::item {
        background: transparent;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 1px;
    }

    QMenuBar::item:selected {
        background: #505050;
    }

    QMenu {
        background-color: #2a2a2a;
        border: 2px solid #555555;
        border-radius: 6px;
        color: #e8e8e8;
        padding: 4px;
    }

    QMenu::item {
        padding: 6px 24px;
        border-radius: 4px;
        margin: 1px;
    }

    QMenu::item:selected {
        background: #505050;
    }

    QToolBar {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #3a3a3a, stop: 1 #2a2a2a);
        border: 1px solid #555555;
        spacing: 2px;
        padding: 4px;
    }

    /* ================================
       ÉLÉMENTS DIVERS
       ================================ */
    QCheckBox, QRadioButton {
        color: #e8e8e8;
        font-weight: 500;
        spacing: 8px;
    }

    QCheckBox::indicator, QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #555555;
        border-radius: 3px;
        background: #2a2a2a;
    }

    QCheckBox::indicator:checked {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #505050, stop: 1 #454545);
        border-color: #707070;
    }

    QRadioButton::indicator {
        border-radius: 8px;
    }

    QProgressBar {
        border: 2px solid #404040;
        border-radius: 6px;
        background: #1a1a1a;
        text-align: center;
        font-weight: 600;
        color: #ffffff;
    }

    QProgressBar::chunk {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #505050, stop: 1 #454545);
        border-radius: 4px;
        margin: 1px;
    }

    QSlider::groove:horizontal {
        border: 1px solid #404040;
        height: 6px;
        background: #2a2a2a;
        margin: 0;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #606060, stop: 1 #505050);
        border: 1px solid #707070;
        width: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }

    /* ================================
       TOOLTIPS
       ================================ */
    QToolTip {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #404040, stop: 1 #353535);
        color: #ffffff;
        border: 1px solid #606060;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 9pt;
        font-weight: 500;
    }


"""

class PandasModel(QAbstractTableModel):
    def __init__(self, data): super().__init__(); self._data = data
    def rowCount(self, parent=None): return self._data.shape[0]
    def columnCount(self, parent=None): return self._data.shape[1]
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole: return str(self._data.iloc[index.row(), index.column()])
        return None
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal: return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical: return str(self._data.index[section])
        return None

class Bridge(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot(list)
    def receive_data(self, data):
        self.parent()._handle_js_data(data)

class MainWindow(QMainWindow):
    """
    La Vue dans une architecture MVC.
    - Contient tout le code de l'interface graphique.
    - Ne contient PAS de logique métier ni d'état de l'application.
    - Reçoit ses ordres du Contrôleur via des signaux.
    - Informe le Contrôleur des actions de l'utilisateur.
    """
    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller

        self.default_features = [
            "median10", "median30", "median50", "std10", "std30", "std50",
            "kurtosis30", "kurtosis50", "skewness30", "skewness50", "slope10", 
            "slope30", "slope50", "slope10_accel", "slope30_accel", "slope50_accel",
            "iqr10", "iqr30", "iqr50", "zscore_120", "range_pos10", "range_pos30", 
            "range_pos50", "volatility_ratio_10_30", "volatility_ratio_10_50", 
            "volatility_ratio_30_50", "time_in_active_state", "energy_in_active_state",
            "log_slope10", "log_slope30", "log_slope50",
            "log_slope10_accel", "log_slope30_accel", "log_slope50_accel"
        ]
        self.sim_vrp_line = None
        self.sim_truth_line = None
        self.sim_pred_line = None
        self.sim_prob_lines = {}
        self.sim_all_dates = []
        self.sim_all_vrp = []
        self.train_df_for_plot = None
        self.val_df_for_plot = None
        self.test_df_for_plot = None
        self.feature_matrix_for_analysis = None
        self.processed_df = None
        self.original_df = None
        self.last_training_results_for_tuning = None 
        self.ephemeral_label_rules = {} 
        self.current_scenario_df = None
        self.bridge = Bridge(self)
    
        self.setWindowTitle("Analyse Prédictive des Événements Volcaniques")
        self.setGeometry(100, 100, 1400, 900)
        
        self.tabs = QTabWidget() 
        self.setCentralWidget(self.tabs)
        
        self.create_data_tab()
        self.create_feature_tab()
        self.create_training_tab()
        self.create_segmentation_tab()
        self.create_comparison_tab()
        self.create_prediction_tab()
        self.create_simulation_tab()
        self.create_xai_tab()
        self.create_scenario_tab()

        self._connect_signals()
        
        self.status_label.setText("Bienvenue ! Veuillez charger un fichier pour commencer.")

    def create_scenario_tab(self):
        self.scenario_tab = QWidget()
        main_layout = QHBoxLayout(self.scenario_tab)

        left_panel = QGroupBox("Pilote du Scénario 'What-If'")
        left_panel.setMaximumWidth(400)
        left_layout = QFormLayout(left_panel)

        self.scenario_model_label = QLabel("Aucun modèle entraîné")
        self.scenario_model_label.setStyleSheet("font-weight: bold;")
        
        self.scenario_data_selector = QComboBox()
        self.scenario_data_selector.addItems(["Utiliser le jeu de test actuel", "Utiliser le jeu de validation actuel"])
        self.analyze_scenario_btn = QPushButton("Analyser ce Scénario")
        self.analyze_scenario_btn.setProperty("class", "primary")
        self.analyze_scenario_btn.setEnabled(False) 

        help_text = QLabel(
            "<b><u>Comment ça marche ?</u></b><br>"
            "1. Utilisez les outils <b>Lasso</b> ou <b>Box Select</b> dans la barre d'outils du graphique ci-dessus.<br>"
            "2. Sélectionnez les points que vous souhaitez modifier.<br>"
            "3. Déplacez les points verticalement pour simuler une nouvelle tendance VRP.<br>"
            "4. Cliquez sur 'Analyser ce Scénario' pour voir l'impact sur les probabilités."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("background-color: #2c2c2c; border: 1px solid #444; border-radius: 4px; padding: 8px;")

        left_layout.addRow("Modèle Actif :", self.scenario_model_label)
        left_layout.addRow("Données de base :", self.scenario_data_selector)
        left_layout.addRow(self.analyze_scenario_btn)
        left_layout.addRow(help_text) 

        right_panel = QGroupBox("Laboratoire de Scénarios")
        right_layout = QVBoxLayout(right_panel)
        self.scenario_editor_canvas = QWebEngineView()
        self.scenario_result_canvas = QWebEngineView()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.scenario_editor_canvas)
        splitter.addWidget(self.scenario_result_canvas)
        splitter.setSizes([400, 300])
        right_layout.addWidget(splitter)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self.tabs.addTab(self.scenario_tab, "Scénarios (What-If)")

    def _add_or_update_ephemeral_label(self):
        label_name = self.new_label_name.text().strip()
        prob_min = self.prob_min_spin.value()
        prob_max = self.prob_max_spin.value()

        if not label_name or prob_min >= prob_max:
            self._show_message_box("Erreur", "Veuillez entrer un nom de label valide et un intervalle de probabilité correct (min < max).")
            return

        self.ephemeral_label_rules[label_name] = (prob_min, prob_max)
        
        self.ephemeral_labels_list.clear()
        self.ephemeral_labels_list.addItems(self.ephemeral_label_rules.keys())

        self._recalculate_and_redraw_all()

    def _remove_ephemeral_label(self):
        selected_items = self.ephemeral_labels_list.selectedItems()
        if not selected_items:
            return
            
        label_to_remove = selected_items[0].text()
        if label_to_remove in self.ephemeral_label_rules:
            del self.ephemeral_label_rules[label_to_remove]

        self.ephemeral_labels_list.clear()
        self.ephemeral_labels_list.addItems(self.ephemeral_label_rules.keys())
        
        self._recalculate_and_redraw_all()

    def _recalculate_and_redraw_all(self):
        """Fonction centrale qui recalcule les prédictions et redessine les graphiques."""
        if not self.last_training_results_for_tuning:
            return

        slider_value = self.threshold_slider.value()
        self._update_predictions_with_threshold(slider_value)
    
    def create_segmentation_tab(self):
        self.seg_tab = QWidget()
        main_layout = QHBoxLayout(self.seg_tab)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(450)

        action_group = QGroupBox("1. Lancement de l'Entraînement")
        action_layout = QVBoxLayout(action_group)
        self.train_segmenter_button = QPushButton("Isoler Blocs Actifs & Entraîner Segmenteur")
        self.train_segmenter_button.setProperty("class", "primary")
        action_layout.addWidget(self.train_segmenter_button)
        
        log_group = QGroupBox("2. Logs de l'Extraction de Blocs")
        log_layout = QVBoxLayout(log_group)
        self.seg_log_text = QTextEdit()
        self.seg_log_text.setReadOnly(True)
        self.seg_log_text.setMinimumHeight(150)
        log_layout.addWidget(self.seg_log_text)

        results_group = QGroupBox("3. Performance du Segmenteur")
        results_layout = QVBoxLayout(results_group)
        self.seg_report_text = QTextEdit()
        self.seg_report_text.setReadOnly(True)
        self.seg_confusion_matrix_canvas = QWebEngineView()
        self.seg_plot_canvas = QWebEngineView()
        results_layout.addWidget(QLabel("Rapport de Classification:"))
        results_layout.addWidget(self.seg_report_text)
        results_layout.addWidget(QLabel("Matrice de Confusion:"))
        results_layout.addWidget(self.seg_confusion_matrix_canvas)

        left_layout.addWidget(action_group)
        left_layout.addWidget(log_group)
        left_layout.addWidget(results_group)
        left_layout.addStretch()

        right_panel = QGroupBox("Analyse Détaillée des Prédictions")
        right_layout = QVBoxLayout(right_panel)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.seg_details_table = QTableWidget()
        self.seg_details_table.setAlternatingRowColors(True)
        self.seg_details_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        v_splitter.addWidget(self.seg_plot_canvas)
        v_splitter.addWidget(self.seg_details_table)
        v_splitter.setSizes([600, 300]) 
        
        right_layout.addWidget(v_splitter)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self.tabs.addTab(self.seg_tab, "4. Segmentation Fine")

    def create_xai_tab(self):
        self.xai_tab = QWidget()
        main_layout = QHBoxLayout(self.xai_tab)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QGroupBox("Analyse d'une Prédiction Spécifique")
        left_layout = QVBoxLayout(left_panel)
        
        self.xai_data_table = QTableWidget()
        self.xai_data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.xai_data_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.explain_button = QPushButton("Expliquer la Prédiction Sélectionnée")
        
        left_layout.addWidget(QLabel("1. Sélectionnez une prédiction dans le jeu de test :"))
        left_layout.addWidget(self.xai_data_table)
        left_layout.addWidget(self.explain_button)
    
        right_panel_group = QGroupBox("Résultats de l'Explication (SHAP)")
        right_panel_layout = QVBoxLayout(right_panel_group)
        right_tabs = QTabWidget()
        local_xai_tab = QWidget()
        local_xai_layout = QVBoxLayout(local_xai_tab)
        
        self.xai_info_label = QLabel("Sélectionnez une prédiction et cliquez sur 'Expliquer'.")
        self.xai_info_label.setWordWrap(True)
        self.xai_info_label.setMaximumHeight(80)
        
        self.xai_waterfall_canvas = QWebEngineView()
        
        local_xai_layout.addWidget(self.xai_info_label)
        local_xai_layout.addWidget(self.xai_waterfall_canvas)
        right_tabs.addTab(local_xai_tab, "Explication Locale")

        global_xai_tab = QWidget()
        global_xai_layout = QFormLayout(global_xai_tab)
        
        self.feature_selector_combo = QComboBox()
        self.interaction_selector_combo = QComboBox()
        self.interaction_selector_combo.addItem("Auto")
        
        self.xai_dependence_canvas = QWebEngineView()
        
        global_xai_layout.addRow("Caractéristique Principale :", self.feature_selector_combo)
        global_xai_layout.addRow("Colorer par (Interaction) :", self.interaction_selector_combo)
        global_xai_layout.addRow(self.xai_dependence_canvas)
        right_tabs.addTab(global_xai_tab, "Analyse de Dépendance (Globale)")
        
        right_panel_layout.addWidget(right_tabs)

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel_group) 
        main_splitter.setSizes([450, 800])
        
        main_layout.addWidget(main_splitter)
    
        self.tabs.addTab(self.xai_tab, "7. Explicabilité (XAI)")

    def create_data_tab(self):
        self.data_tab = QWidget()
        main_layout = QVBoxLayout(self.data_tab)
        main_layout.setSpacing(8)  
        main_layout.setContentsMargins(10, 10, 10, 10)  
        
        top_widget = QWidget()
        top_widget.setMaximumHeight(50)  
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_button = QPushButton(" Charger Fichier")
        self.load_button.setMaximumWidth(150)  
        self.status_label = QLabel("Prêt à charger des données...")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.status_label, 1)  
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)  
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)  
        
        table_label = QLabel("Données Chargées")
        table_label.setStyleSheet("font-weight: bold; color: #333;")
        left_layout.addWidget(table_label)
        
        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setMinimumWidth(350)  
        left_layout.addWidget(self.table_view)
        
        main_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)  
        right_layout.setSpacing(8)
        
        graph_group = QGroupBox("Aperçu des Données")
        graph_group.setMinimumHeight(250)  
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.setContentsMargins(8, 15, 8, 8) 
        
        self.overview_canvas = QWebEngineView()
        self.overview_canvas.setMinimumHeight(200)
        graph_layout.addWidget(self.overview_canvas)
        
        stats_group = QGroupBox("Statistiques Descriptives")
        stats_group.setMinimumHeight(150)
        stats_group.setMaximumHeight(200)  
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 15, 8, 8)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(160)
        self.stats_text.setStyleSheet(
            "background-color: #2b2b2b; "
            "color: #ffffff; "
            "border: 1px solid #555; "
            "font-family: 'Consolas', 'Monaco', monospace; "
            "font-size: 11px;"
        )
        stats_layout.addWidget(self.stats_text)
        
        right_layout.addWidget(graph_group, 2)  # 2/3 de l'espace
        right_layout.addWidget(stats_group, 1)   # 1/3 de l'espace
        
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 550]) 
        
        bottom_widget = QWidget()
        bottom_widget.setMaximumHeight(120)  
        config_layout = QHBoxLayout(bottom_widget)
        config_layout.setSpacing(12)
        config_layout.setContentsMargins(0, 8, 0, 0)
        
        balancing_groupbox = QGroupBox("Équilibrage (Ancienne Méthode)")
        balancing_groupbox.setMaximumWidth(280)  
        form_layout = QVBoxLayout(balancing_groupbox)  
        form_layout.setSpacing(5)
        form_layout.setContentsMargins(8, 15, 8, 8)
        
        self.cb_bal = QCheckBox("Activer pour entraînement")
        self.btn_bal = QPushButton("Générer un aperçu équilibré")
        self.btn_bal.setMaximumHeight(30)
        
        form_layout.addWidget(self.cb_bal)
        form_layout.addWidget(self.btn_bal)
        form_layout.addStretch() 
        
        self.target_groupbox = QGroupBox("Définition de la Cible pour l'Entraînement")
        target_layout = QVBoxLayout(self.target_groupbox)
        target_layout.setSpacing(4)
        target_layout.setContentsMargins(8, 15, 8, 8)
        
        self.rb_binary_mode = QRadioButton("Classification Binaire (Mode Détecteur)")
        self.rb_binary_mode.setToolTip("Approche recommandée : 'Calm' vs 'Actif'")
        
        self.rb_multiclass_mode = QRadioButton("Classification Multi-Classes (4 états)")
        self.rb_multiclass_mode.setToolTip("Approche avancée : Segmentation fine des événements")
        self.rb_multiclass_mode.setChecked(True)
        
        self.multiclass_options_widget = QWidget()
        multiclass_layout = QHBoxLayout(self.multiclass_options_widget)
        multiclass_layout.setContentsMargins(15, 0, 0, 0)
        multiclass_layout.setSpacing(10)
    
        pre_label = QLabel("Précurseurs:")
        self.pre_event_ratio_spin = QDoubleSpinBox()
        self.pre_event_ratio_spin.setRange(0.0, 1.0)
        self.pre_event_ratio_spin.setSingleStep(0.1)
        self.pre_event_ratio_spin.setValue(0.3)
        self.pre_event_ratio_spin.setMaximumWidth(70)
        
        post_label = QLabel("Relaxation:")
        self.post_event_ratio_spin = QDoubleSpinBox()
        self.post_event_ratio_spin.setRange(0.0, 1.0)
        self.post_event_ratio_spin.setSingleStep(0.1)
        self.post_event_ratio_spin.setValue(0.2)
        self.post_event_ratio_spin.setMaximumWidth(70)
        
        multiclass_layout.addWidget(pre_label)
        multiclass_layout.addWidget(self.pre_event_ratio_spin)
        multiclass_layout.addWidget(post_label)
        multiclass_layout.addWidget(self.post_event_ratio_spin)
        multiclass_layout.addStretch()
        
        target_layout.addWidget(self.rb_binary_mode)
        target_layout.addWidget(self.rb_multiclass_mode)
        target_layout.addWidget(self.multiclass_options_widget)
        
        def on_target_mode_changed():
            self.multiclass_options_widget.setVisible(self.rb_multiclass_mode.isChecked())
        
        self.rb_binary_mode.toggled.connect(on_target_mode_changed)
        self.rb_multiclass_mode.toggled.connect(on_target_mode_changed)
        
        config_layout.addWidget(balancing_groupbox, 0)  
        config_layout.addWidget(self.target_groupbox, 1)  
    
        main_layout.addWidget(top_widget, 0)      
        main_layout.addWidget(main_splitter, 1)   
        main_layout.addWidget(bottom_widget, 0)   
        
        self.tabs.addTab(self.data_tab, "1. Gestion Données")
    def create_feature_tab(self):
        self.feature_tab = QWidget()
        main_layout = QVBoxLayout(self.feature_tab)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        win_group = QGroupBox("Configuration des Fenêtres")
        win_layout = QFormLayout(win_group)
        self.win10_spin = QSpinBox(); self.win10_spin.setRange(1, 200); self.win10_spin.setValue(10)
        self.win30_spin = QSpinBox(); self.win30_spin.setRange(1, 200); self.win30_spin.setValue(30)
        self.win50_spin = QSpinBox(); self.win50_spin.setRange(1, 200); self.win50_spin.setValue(50)
        win_layout.addRow("Taille Fenêtre '...10':", self.win10_spin)
        win_layout.addRow("Taille Fenêtre '...30':", self.win30_spin)
        win_layout.addRow("Taille Fenêtre '...50':", self.win50_spin)

        features_group = QGroupBox("Caractéristiques à Générer")
        features_layout = QVBoxLayout(features_group)
        self.feature_list_widget = QListWidget()
        
        self.feature_list_widget.addItems(self.default_features)
        self.feature_list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
       
        for i in range(self.feature_list_widget.count()): 
            self.feature_list_widget.item(i).setSelected(True)
        features_layout.addWidget(self.feature_list_widget)

 
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

   
        self.select_all_cb = QCheckBox("Tout sélectionner / désélectionner")
        self.select_all_cb.setChecked(True)
        
    
        self.restore_features_btn = QPushButton("Restaurer la liste complète")

        self.select_nn_preset_btn = QPushButton("Preset pour Réseau de Neurones (NN)") # 
        self.select_nn_preset_btn.setToolTip("Sélectionne un sous-ensemble de features robustes, idéal pour les NN.")

        actions_layout.addWidget(self.select_nn_preset_btn)
        

       
        top_n_layout = QHBoxLayout()
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, len(self.default_features))
        self.top_n_spin.setValue(10)
        self.select_top_n_btn = QPushButton("Sélectionner les N Meilleures")
        self.select_top_n_btn.setToolTip(
            "Nécessite qu'un modèle ait été entraîné (onglet 3) pour connaître l'importance des caractéristiques."
        )
        top_n_layout.addWidget(self.top_n_spin)
        top_n_layout.addWidget(self.select_top_n_btn)

      
        self.norm_cb = QCheckBox("Normaliser les caractéristiques (pour entraînement)")
        self.norm_cb.setChecked(True)
        self.gen_btn = QPushButton("Générer et Analyser les Caractéristiques")
        
     
        actions_layout.addWidget(self.select_all_cb)
        actions_layout.addWidget(self.restore_features_btn)
        actions_layout.addLayout(top_n_layout)
        actions_layout.addSpacing(15) 
        actions_layout.addWidget(self.norm_cb)
        actions_layout.addWidget(self.gen_btn)

        actions_layout.addWidget(self.select_nn_preset_btn) 


       
        left_layout.addWidget(win_group)
        left_layout.addWidget(features_group)
        left_layout.addWidget(actions_group)

        right_panel = QWidget(); right_layout = QHBoxLayout(right_panel)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        indiv_group = QGroupBox("Analyse de la Caractéristique Sélectionnée"); indiv_layout = QVBoxLayout(indiv_group)
        self.feature_dist_canvas = QWebEngineView()
        self.feature_boxplot_canvas = QWebEngineView()


        indiv_layout.addWidget(self.feature_dist_canvas); indiv_layout.addWidget(self.feature_boxplot_canvas)
        v_splitter.addWidget(indiv_group)
        corr_group = QGroupBox("Matrice de Corrélation Globale"); corr_layout = QVBoxLayout(corr_group)
        self.corr_matrix_canvas = QWebEngineView() 
        corr_layout.addWidget(self.corr_matrix_canvas)
        v_splitter.addWidget(corr_group)
        right_layout.addWidget(v_splitter); right_panel.setLayout(right_layout)

        h_splitter.addWidget(left_panel); h_splitter.addWidget(right_panel)
        h_splitter.setSizes([300, 700])
        main_layout.addWidget(h_splitter)
        self.tabs.addTab(self.feature_tab, "2. Extraction Caractéristiques")

    def create_training_tab(self):
        self.training_tab = QWidget()
        main_layout = QHBoxLayout(self.training_tab)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        model_selection_group = QGroupBox("1. Sélection du Modèle"); model_selection_layout = QFormLayout(model_selection_group)
        self.model_type_combo = QComboBox(); self.model_type_combo.addItems(["Neural Network", "Random Forest", "LightGBM", "K-Nearest Neighbors (KNN)"]); model_selection_layout.addRow("Algorithme :", self.model_type_combo)
        split_config_group = QGroupBox("2. Configuration de la Partition"); split_form = QFormLayout(split_config_group)
        self.split_type_combo = QComboBox(); self.split_type_combo.addItems(["Temporel (Recommandé)", "Aléatoire Stratifié"]); split_form.addRow("Type de Partition:", self.split_type_combo)
        self.train_ratio_spin = QSpinBox(); self.train_ratio_spin.setRange(1, 100); self.train_ratio_spin.setValue(60); split_form.addRow("Ratio Entraînement (%)", self.train_ratio_spin)
        self.val_ratio_spin = QSpinBox(); self.val_ratio_spin.setRange(1, 100); self.val_ratio_spin.setValue(20); split_form.addRow("Ratio Validation (%)", self.val_ratio_spin)
        self.test_ratio_spin = QSpinBox(); self.test_ratio_spin.setRange(1, 100); self.test_ratio_spin.setValue(20); split_form.addRow("Ratio Test (%)", self.test_ratio_spin)
        params_group = QGroupBox("3. Hyperparamètres du Modèle"); params_layout = QVBoxLayout(params_group); self.params_stack = QStackedWidget(); params_layout.addWidget(self.params_stack)
        nn_panel = QWidget(); nn_form = QFormLayout(nn_panel)
        self.nn_epochs_spin = QSpinBox(); self.nn_epochs_spin.setRange(10, 1000); self.nn_epochs_spin.setValue(200); nn_form.addRow("Epochs (max) :", self.nn_epochs_spin)
        self.nn_batch_size_spin = QSpinBox(); self.nn_batch_size_spin.setRange(8, 256); self.nn_batch_size_spin.setValue(64); self.nn_batch_size_spin.setSingleStep(8); nn_form.addRow("Taille du Batch :", self.nn_batch_size_spin)
        self.nn_lr_spin = QDoubleSpinBox(); self.nn_lr_spin.setRange(0.0001, 0.1); self.nn_lr_spin.setValue(0.001); self.nn_lr_spin.setDecimals(4); self.nn_lr_spin.setSingleStep(0.0001); nn_form.addRow("Taux d'Apprentissage :", self.nn_lr_spin)
        rf_panel = QWidget(); rf_form = QFormLayout(rf_panel)
        self.rf_trees_spin = QSpinBox(); self.rf_trees_spin.setRange(10, 1000); self.rf_trees_spin.setValue(200); self.rf_trees_spin.setSingleStep(10); rf_form.addRow("Nombre d'arbres (max):", self.rf_trees_spin)
        self.rf_max_depth_spin = QSpinBox(); self.rf_max_depth_spin.setRange(5, 50); self.rf_max_depth_spin.setValue(15); rf_form.addRow("Profondeur max :", self.rf_max_depth_spin)
        self.rf_min_leaf_spin = QSpinBox(); self.rf_min_leaf_spin.setRange(1, 20); self.rf_min_leaf_spin.setValue(1); rf_form.addRow("Min samples par feuille :", self.rf_min_leaf_spin)
        self.rf_class_weight_cb = QCheckBox("Utiliser 'class_weight=balanced'"); self.rf_class_weight_cb.setChecked(True); rf_form.addRow(self.rf_class_weight_cb)
        lgbm_panel = QWidget(); lgbm_layout = QFormLayout(lgbm_panel); lgbm_info_label = QLabel("La recherche d'hyperparamètres pour LightGBM est\nautomatisée."); lgbm_info_label.setWordWrap(True); lgbm_layout.addRow(lgbm_info_label)
        knn_panel = QWidget(); knn_form = QFormLayout(knn_panel)
        self.max_k_spin = QSpinBox(); self.max_k_spin.setRange(1, 100); self.max_k_spin.setValue(30); knn_form.addRow("Max k pour KNN:", self.max_k_spin)
        self.weights_uniform_cb = QCheckBox("uniform"); self.weights_uniform_cb.setChecked(True)
        self.weights_distance_cb = QCheckBox("distance"); self.weights_distance_cb.setChecked(True)
        weights_layout = QHBoxLayout(); weights_layout.addWidget(self.weights_uniform_cb); weights_layout.addWidget(self.weights_distance_cb); knn_form.addRow("KNN Weights:", weights_layout)
        self.metric_euclidean_cb = QCheckBox("minkowski"); self.metric_euclidean_cb.setChecked(True)
        self.metric_manhattan_cb = QCheckBox("manhattan"); self.metric_manhattan_cb.setChecked(False)
        metrics_layout = QHBoxLayout(); metrics_layout.addWidget(self.metric_euclidean_cb); metrics_layout.addWidget(self.metric_manhattan_cb); knn_form.addRow("KNN Metrics:", metrics_layout)
        self.params_stack.addWidget(nn_panel); self.params_stack.addWidget(rf_panel); self.params_stack.addWidget(lgbm_panel); self.params_stack.addWidget(knn_panel)
        weighting_group = QGroupBox("4. Pondération Temporelle (Avancé)")
        weighting_layout = QFormLayout(weighting_group)
        self.cb_sample_weight = QCheckBox("Donner plus de poids aux points récents"); self.cb_sample_weight.setChecked(True)
        self.decay_rate_spin = QDoubleSpinBox(); self.decay_rate_spin.setRange(0.001, 0.1); self.decay_rate_spin.setSingleStep(0.001); self.decay_rate_spin.setDecimals(3); self.decay_rate_spin.setValue(0.005)
        self.cb_sample_weight.toggled.connect(self.decay_rate_spin.setEnabled); weighting_layout.addRow(self.cb_sample_weight); weighting_layout.addRow("Taux de décroissance :", self.decay_rate_spin)
        buttons_layout = QHBoxLayout()
        self.run_training_button = QPushButton("LANCER ENTRAÎNEMENT"); self.clear_button = QPushButton("Nettoyer"); buttons_layout.addWidget(self.run_training_button); buttons_layout.addWidget(self.clear_button)
        results_group = QGroupBox("5. Résultats"); results_layout = QVBoxLayout(results_group); self.metrics_text = QTextEdit(); self.metrics_text.setReadOnly(True); results_layout.addWidget(QLabel("Rapport de Classification Détaillé:")); results_layout.addWidget(self.metrics_text)
        save_group = QGroupBox("Sauvegarder le Modèle"); save_layout = QFormLayout(save_group); self.model_name_input = QLineEdit(); self.save_model_button = QPushButton("Sauvegarder"); self.save_model_button.setEnabled(False); save_layout.addRow("Nom du Modèle:", self.model_name_input); save_layout.addRow(self.save_model_button)
        left_layout.addWidget(model_selection_group); left_layout.addWidget(split_config_group); left_layout.addWidget(params_group); left_layout.addWidget(weighting_group); left_layout.addLayout(buttons_layout); left_layout.addWidget(results_group); left_layout.addWidget(save_group); left_layout.addStretch()

        
     
        right_tabs = QTabWidget()

        self.main_plot_canvas = QWebEngineView()
        self.metrics_plot_canvas = QWebEngineView()
        self.probability_evolution_canvas = QWebEngineView()
        self.confusion_matrix_canvas = QWebEngineView()
        self.feature_importance_canvas = QWebEngineView()
        self.permutation_importance_canvas = QWebEngineView() 
        self.learning_curves_canvas = QWebEngineView()

        plot_tab = QWidget(); plot_layout = QVBoxLayout(plot_tab); plot_layout.addWidget(self.main_plot_canvas)
        right_tabs.addTab(plot_tab, "Prédictions vs Réalité")

        metrics_tab = QWidget(); metrics_layout = QVBoxLayout(metrics_tab); metrics_layout.addWidget(self.metrics_plot_canvas)
        right_tabs.addTab(metrics_tab, "Métriques (Indexes)") 

        prob_tab = QWidget(); prob_layout = QVBoxLayout(prob_tab)
        prob_layout.addWidget(self.probability_evolution_canvas)
        
        threshold_group = QGroupBox("Ajustement du Seuil de Décision")
        threshold_layout = QFormLayout(threshold_group)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 99); self.threshold_slider.setValue(50)
        self.threshold_slider.setEnabled(False)
        self.threshold_value_label = QLabel("Seuil : 0.50")
        self.threshold_metrics_label = QLabel("Précision: N/A | Rappel: N/A")
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.threshold_slider)
        slider_layout.addWidget(self.threshold_value_label)
        
        threshold_layout.addRow(slider_layout)
        threshold_layout.addRow(self.threshold_metrics_label)
        prob_layout.addWidget(threshold_group)

        ephemeral_group = QGroupBox("Laboratoire de Segmentation par Probabilité")
        ephemeral_layout = QFormLayout(ephemeral_group)
        
        self.new_label_name = QLineEdit("Précurseur_Faible")
        self.prob_min_spin = QDoubleSpinBox(); self.prob_min_spin.setRange(0.0, 1.0); self.prob_min_spin.setValue(0.51)
        self.prob_max_spin = QDoubleSpinBox(); self.prob_max_spin.setRange(0.0, 1.0); self.prob_max_spin.setValue(0.70)
        self.add_ephemeral_label_btn = QPushButton("Ajouter/Mettre à jour le Label Éphémère")

        self.ephemeral_labels_list = QListWidget()
        self.remove_ephemeral_label_btn = QPushButton("Supprimer le Label Sélectionné")
        
        ephemeral_layout.addRow("Nom du nouveau label :", self.new_label_name)
        prob_range_layout = QHBoxLayout()
        prob_range_layout.addWidget(QLabel("Si P(Actif) est entre"))
        prob_range_layout.addWidget(self.prob_min_spin)
        prob_range_layout.addWidget(QLabel("et"))
        prob_range_layout.addWidget(self.prob_max_spin)
        ephemeral_layout.addRow(prob_range_layout)
        ephemeral_layout.addRow(self.add_ephemeral_label_btn)
        ephemeral_layout.addRow("Labels Éphémères Actifs :", self.ephemeral_labels_list)
        ephemeral_layout.addRow(self.remove_ephemeral_label_btn)
        
        prob_layout.addWidget(ephemeral_group)
        
        right_tabs.addTab(prob_tab, "Évolution des Probabilités")
        
        cm_tab = QWidget(); cm_layout = QVBoxLayout(cm_tab); cm_layout.addWidget(self.confusion_matrix_canvas)
        right_tabs.addTab(cm_tab, "Matrice de Confusion")

        fi_tab = QWidget(); fi_layout = QVBoxLayout(fi_tab); fi_layout.addWidget(self.feature_importance_canvas)
        right_tabs.addTab(fi_tab, "Importance Caractéristiques")

        pi_tab = QWidget(); pi_layout = QVBoxLayout(pi_tab); pi_layout.addWidget(self.permutation_importance_canvas)
        right_tabs.addTab(pi_tab, "Importance (Permutation)") 


        lc_tab = QWidget(); lc_layout = QVBoxLayout(lc_tab); lc_layout.addWidget(self.learning_curves_canvas)
        right_tabs.addTab(lc_tab, "Courbes d'Apprentissage")
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_tabs, 2)
        self.tabs.addTab(self.training_tab, "3. Entraînement")


   

    def update_permutation_importance_plot(self, results):
        """
        Calcule et affiche l'importance des caractéristiques par permutation.
        VERSION DÉFINITIVE v3 : Gère correctement pos_label avec les labels textuels.
        """
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import f1_score, make_scorer

        model = results.get('model')
        plot_data = results.get('plot_data')
        
        if not (model and plot_data and self.test_df_for_plot is not None and 'y_test' in plot_data):
            self._clear_plotly_view(self.permutation_importance_canvas, "Données insuffisantes pour le calcul.")
            return

        y_test = plot_data.get('y_test')
        if y_test.empty or len(y_test.unique()) < 2:
            self._clear_plotly_view(self.permutation_importance_canvas, "Jeu de test vide ou ne contenant qu'une seule classe.")
            return
            
        if self.feature_matrix_for_analysis is None:
             self._clear_plotly_view(self.permutation_importance_canvas, "La matrice des caractéristiques n'est pas disponible.")
             return

        model_feature_names = model.feature_names_in_
        X_test = self.feature_matrix_for_analysis.loc[y_test.index].reindex(columns=model_feature_names, fill_value=0)

     
        positive_class_name = next((label for label in y_test.unique() if label != 'Calm'), None)
        
        if not positive_class_name:
            self._clear_plotly_view(self.permutation_importance_canvas, "Impossible de trouver une classe positive ('Actif', 'Pre-Event'...)")
            return
            
      
        scorer_to_use = make_scorer(f1_score, pos_label=positive_class_name, zero_division=0)
        title = f"Permutation Feature Importance (Drop in F1-Sore for Active class) '{positive_class_name}')"
    
        try:
            self.status_label.setText(f"Calcul de l'importance par permutation (cible: {positive_class_name})...")
            QApplication.processEvents()

            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, 
                scoring=scorer_to_use,
                n_jobs=-1
            )
            
            self.status_label.setText("Calcul terminé.")
            
            sorted_idx = perm_result.importances_mean.argsort()
            
            fig = go.Figure(go.Bar(
                x=perm_result.importances_mean[sorted_idx],
                y=X_test.columns[sorted_idx],
                orientation='h',
                error_x=dict(type='data', array=perm_result.importances_std[sorted_idx])
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Baisse de Performance",
                template='plotly_dark'
            )
            self._display_plotly_fig(self.permutation_importance_canvas, fig)

        except Exception as e:
            error_message = f"Erreur lors du calcul de l'importance par permutation:\n{e}"
            self._clear_plotly_view(self.permutation_importance_canvas, error_message)
            import traceback
            traceback.print_exc()

   

    def update_dependence_plot(self):
        """
        Génère et affiche un SHAP Dependence Plot pour la feature sélectionnée.
        VERSION DÉFINITIVE v3 : Gère correctement la conversion des types NumPy.
        """
        if not self.controller.current_explainer:
            self._clear_plotly_view(self.xai_dependence_canvas, "Explainer SHAP non initialisé.")
            return

        feature = self.feature_selector_combo.currentText()
        interaction_feature = self.interaction_selector_combo.currentText()
        
        if not feature:
            return

        shap_values = self.controller.current_explainer.shap_values
        X_test = self.controller.current_explainer.X_test_for_shap
        
        if isinstance(shap_values, list):
            class_names = self.controller.current_explainer.model.classes_
            positive_class_index = list(class_names).index(next(c for c in class_names if c != 'Calm'))
            shap_values_positive = shap_values[positive_class_index]
        else:
            shap_values_positive = shap_values
        
        feature_index = list(X_test.columns).index(feature)

        fig = go.Figure()

        interaction_values = None
        color_bar_title = None
        if interaction_feature != "Auto" and interaction_feature in X_test.columns:
            interaction_values = X_test[interaction_feature]
            color_bar_title = f"Valeur de '{interaction_feature}'"
        else:
            interaction_values = shap_values_positive[:, feature_index]
            color_bar_title = "Impact sur la prédiction (SHAP)"
        
        text_values = np.ravel(interaction_values).tolist()

        fig.add_trace(go.Scatter(
            x=X_test[feature],
            y=shap_values_positive[:, feature_index],
            mode='markers',
            marker=dict(
                color=interaction_values,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title=color_bar_title)
            ),
            text=[f"Interaction: {val:.2f}" for val in text_values],
            hoverinfo='x+y+text'
        ))
        
        fig.update_layout(
            title=f"Relation entre '{feature}' et son Impact sur la Prédiction 'Actif'",
            xaxis_title=f"Valeur de la Caractéristique '{feature}'",
            yaxis_title=f"Valeur SHAP (Impact sur la probabilité)",
            template='plotly_dark'
        )

        self._display_plotly_fig(self.xai_dependence_canvas, fig)

    def update_probability_evolution_plot(self, results):
        if 'probabilities' not in results or 'plot_data' not in results or self.test_df_for_plot is None: return
        
        prob_df = pd.DataFrame(results['probabilities'], index=results['plot_data']['y_test'].index, columns=results['class_names'])
        
        fig = make_subplots(rows=len(prob_df.columns), cols=1, shared_xaxes=True, subplot_titles=prob_df.columns)
        
        color_map = {'Calm': 'royalblue', 'Actif': 'orangered', 'Pre-Event': 'gold', 'High-Paroxysm': 'red', 'Post-Event': 'skyblue'}
        
        for i, class_name in enumerate(prob_df.columns):
            fig.add_trace(go.Scatter(x=prob_df.index, y=prob_df[class_name], name=f'P({class_name})', line=dict(color=color_map.get(class_name, 'white'))), row=i+1, col=1)

        fig.update_layout(title="Évolution Temporelle des Probabilités", template='plotly_dark', showlegend=False)
        self._display_plotly_fig(self.probability_evolution_canvas, fig)
    def update_feature_importance_plot(self, results):
        model = results.get('model')
        if not (model and hasattr(model, 'feature_importances_')): return
        
        importances = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values()
        fig = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation='h'))
        fig.update_layout(title="Feature Importances", template='plotly_dark')
        self._display_plotly_fig(self.feature_importance_canvas, fig)

    def update_metrics_plot(self, results):
        """
        Génère un graphique à barres pour les métriques de performance clés.
        """
        report = results.get('report')
        if not report:
            self._clear_plotly_view(self.metrics_plot_canvas, "Rapport de classification non disponible.")
            return

        class_names = [name for name in report.keys() if name not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        fig = go.Figure()

        for metric in metrics:
            values = [report[name].get(metric, 0) for name in class_names]
            fig.add_trace(go.Bar(
                x=class_names,
                y=values,
                name=metric.title(),
                text=[f'{v:.2f}' for v in values],
                textposition='auto'
            ))

        macro_f1 = report.get('macro avg', {}).get('f1-score', 0)
        
        fig.update_layout(
            barmode='group',
            title=f"Class-wise performance metrics<br><sup>Macro F1-Score Global: {macro_f1:.3f}</sup>",
            xaxis_title="Classe",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            template='plotly_dark',
            legend_title_text='Métrique'
        )

        self._display_plotly_fig(self.metrics_plot_canvas, fig)


    def update_probability_plots(self, results):
        """Met à jour les deux graphiques de l'onglet d'analyse des probabilités."""
        ax1 = self.prob_line_canvas.axes
        ax2 = self.prob_stack_canvas.axes
        ax1.clear()
        ax2.clear()

        if 'probabilities' not in results or 'plot_data' not in results:
            ax1.text(0.5, 0.5, 'Données de probabilité non disponibles.', ha='center', va='center')
            ax2.text(0.5, 0.5, 'Données de probabilité non disponibles.', ha='center', va='center')
            self.prob_line_canvas.draw()
            self.prob_stack_canvas.draw()
            return
            
        y_test = results['plot_data']['y_test']
        probabilities = results['probabilities']
        class_names = results['class_names']
        dates = y_test.index

        for i, class_name in enumerate(class_names):
            ax1.plot(dates, probabilities[:, i], label=f'P({class_name})', lw=2)
        
        ax1.set_title("Temporal Evolution of Probabilities")
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        self.prob_line_canvas.figure.tight_layout()
        self.prob_line_canvas.draw()

        ax2.stackplot(dates, probabilities.T, labels=class_names, alpha=0.8)
        
        ax2.set_title("Composition Cumulative des Probabilités")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        
        self.prob_stack_canvas.figure.tight_layout()
        self.prob_stack_canvas.draw()
    def create_comparison_tab(self):
        self.comp_tab = QWidget(); main_layout = QVBoxLayout(self.comp_tab)
        control_group = QGroupBox("Outils de Comparaison"); control_layout = QHBoxLayout(control_group)
        self.model_a_combo=QComboBox(); self.model_b_combo=QComboBox()
        self.compare_button=QPushButton("Comparer A vs B"); self.leaderboard_button=QPushButton("Afficher Classement")
        form_a=QFormLayout(); form_a.addRow("Modèle A:", self.model_a_combo); form_b=QFormLayout(); form_b.addRow("Modèle B:", self.model_b_combo)
        control_layout.addLayout(form_a); control_layout.addLayout(form_b); control_layout.addWidget(self.compare_button); control_layout.addWidget(self.leaderboard_button)
        self.comparison_tabs = QTabWidget(); self.avsb_tab = QWidget(); avsb_layout = QHBoxLayout(self.avsb_tab)
        self.comparison_table = QTableWidget(); self.radar_canvas = QWebEngineView()
        avsb_layout.addWidget(self.comparison_table, 1); avsb_layout.addWidget(self.radar_canvas, 1)
        self.leaderboard_tab = QWidget(); leaderboard_layout = QVBoxLayout(self.leaderboard_tab); self.leaderboard_table = QTableWidget()
        self.leaderboard_table.setSortingEnabled(True); leaderboard_layout.addWidget(self.leaderboard_table)
        self.comparison_tabs.addTab(self.avsb_tab, "Comparaison Détaillée A vs B"); self.comparison_tabs.addTab(self.leaderboard_tab, "Classement (Ladder)")
        main_layout.addWidget(control_group); main_layout.addWidget(self.comparison_tabs)
        self.tabs.addTab(self.comp_tab, "4. Comparaison")
    
    def create_prediction_tab(self):
        self.pred_tab = QWidget()
        main_layout = QHBoxLayout(self.pred_tab)

        left_panel = QGroupBox("Piloter la Prédiction")
        left_panel.setMaximumWidth(350)
        control_layout = QVBoxLayout(left_panel) 

        base_config_group = QGroupBox("Configuration")
        base_config_layout = QFormLayout(base_config_group)
        self.pred_model_combo = QComboBox()
        self.pred_load_data_button = QPushButton("Charger Données Externes")
        base_config_layout.addRow("Modèle à tester:", self.pred_model_combo)
        base_config_layout.addRow(self.pred_load_data_button)

        replicate_group = QGroupBox("Répliquer un Set de Test")
        replicate_group.setCheckable(True)  
        replicate_group.setChecked(False)
        replicate_layout = QFormLayout(replicate_group)
        self.pred_percentage_spin = QSpinBox()
        self.pred_percentage_spin.setRange(1, 100)
        self.pred_percentage_spin.setValue(20) 
        self.pred_percentage_spin.setSuffix(" %")
        replicate_layout.addRow("Utiliser les derniers... ", self.pred_percentage_spin)
 
        self.pred_replicate_group = replicate_group

   
        self.pred_run_button = QPushButton("LANCER PRÉDICTION")
        self.pred_run_button.setProperty("class", "primary")

      
        control_layout.addWidget(base_config_group)
        control_layout.addWidget(self.pred_replicate_group)
        control_layout.addStretch()
        control_layout.addWidget(self.pred_run_button)
        left_panel.setLayout(control_layout)


        right_panel = QGroupBox("Analyse de la Performance sur Données Externes")
        right_layout = QVBoxLayout(right_panel)
        self.pred_kpi_widget = QWidget()
        self.pred_kpi_layout = QHBoxLayout(self.pred_kpi_widget)
        right_layout.addWidget(self.pred_kpi_widget)
        self.pred_results_tabs = QTabWidget()
        right_layout.addWidget(self.pred_results_tabs)
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.pred_results_canvas = QWebEngineView()
   
        plot_layout.addWidget(self.pred_results_canvas)
        self.pred_results_tabs.addTab(plot_tab, "📈 Vue Temporelle")
        cm_tab = QWidget()
        cm_layout = QVBoxLayout(cm_tab)
        self.pred_confusion_matrix_canvas = QWebEngineView()
        cm_layout.addWidget(self.pred_confusion_matrix_canvas)
        self.pred_results_tabs.addTab(cm_tab, "🔢 Matrice de Confusion")
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        self.pred_report_text = QTextEdit()
        self.pred_report_text.setReadOnly(True)
        report_layout.addWidget(self.pred_report_text)
        self.pred_results_tabs.addTab(report_tab, "📄 Rapport Détaillé")

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.tabs.addTab(self.pred_tab, "5. Prédiction Externe")
    def create_simulation_tab(self):
        self.sim_tab = QWidget()
        main_layout = QHBoxLayout(self.sim_tab)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)

        model_group = QGroupBox("1. Sélection du Modèle")
        model_layout = QFormLayout(model_group)
        self.sim_model_combo = QComboBox()
        model_layout.addRow("Modèle à utiliser:", self.sim_model_combo)

        gen_group = QGroupBox("2. Configuration des Données Synthétiques")
        gen_layout = QFormLayout(gen_group)
        self.sim_duration_spin = QSpinBox()
        self.sim_duration_spin.setRange(1, 365 * 5); self.sim_duration_spin.setValue(30)
        self.sim_events_spin = QSpinBox()
        self.sim_events_spin.setRange(0, 100); self.sim_events_spin.setValue(5)
        self.sim_ratio_long_spin = QDoubleSpinBox()
        self.sim_ratio_long_spin.setRange(0.0, 1.0); self.sim_ratio_long_spin.setValue(0.2); self.sim_ratio_long_spin.setSingleStep(0.1)
        gen_layout.addRow("Durée (jours):", self.sim_duration_spin)
        gen_layout.addRow("Nombre d'événements:", self.sim_events_spin)
        gen_layout.addRow("Ratio d'événements longs:", self.sim_ratio_long_spin)

        control_group = QGroupBox("3. Contrôles de la Simulation")
        control_layout = QHBoxLayout(control_group)
        self.sim_start_button = QPushButton("DÉMARRER")
        self.sim_pause_button = QPushButton("PAUSE")
        self.sim_stop_button = QPushButton("ARRÊTER")
        self.sim_pause_button.setEnabled(False)
        self.sim_stop_button.setEnabled(False)
        control_layout.addWidget(self.sim_start_button)
        control_layout.addWidget(self.sim_pause_button)
        control_layout.addWidget(self.sim_stop_button)

        left_layout.addWidget(model_group)
        left_layout.addWidget(gen_group)
        left_layout.addWidget(control_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        vrp_plot_group = QGroupBox("Évolution Temporelle VRP et Prédictions")
        vrp_plot_layout = QVBoxLayout(vrp_plot_group)
        self.simulation_vrp_canvas = QWebEngineView()
        vrp_plot_layout.addWidget(self.simulation_vrp_canvas)
        
        prob_plot_group = QGroupBox("Probabilités des États en Temps Réel")
        prob_plot_layout = QVBoxLayout(prob_plot_group)
        self.simulation_prob_canvas = QWebEngineView()
        prob_plot_layout.addWidget(self.simulation_prob_canvas)

        right_layout.addWidget(vrp_plot_group)
        right_layout.addWidget(prob_plot_group)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.tabs.addTab(self.sim_tab, "6. Simulation Temps Réel")



    def _trigger_start_simulation(self):
        params = {
            'model_name': self.sim_model_combo.currentText(),
            'duration_days': self.sim_duration_spin.value(),
            'num_events': self.sim_events_spin.value(),
            'ratio_long_events': self.sim_ratio_long_spin.value()
        }

        worker = self.controller.start_simulation(params)
        if worker:
            dialog = ProcessingDialog("Préparation de la Simulation", self)
            worker.progress_updated.connect(dialog.update_progress)
            dialog.cancelled.connect(worker.cancel)
            worker.finished.connect(dialog.task_finished)
            dialog.exec()
    


    def _handle_js_data(self, data):
        """Reçoit les données modifiées du JS et lance l'analyse."""
        print("Données modifiées reçues de JavaScript !")
        if self.current_scenario_df is not None:
            modified_df = self.current_scenario_df.copy()
            modified_df['VRP'] = data
            model_name = self.scenario_model_label.text()
            self.controller.run_whatif_analysis(model_name, modified_df)


    @pyqtSlot()
    def _setup_scenario_editor(self):
        if self.last_training_results_for_tuning is None:
            self._clear_plotly_view(self.scenario_editor_canvas, "Veuillez d'abord entraîner un modèle.")
            return

        source_choice = self.scenario_data_selector.currentText()
        if "test" in source_choice:
            self.current_scenario_df = self.test_df_for_plot.copy()
        else:
            self.current_scenario_df = self.val_df_for_plot.copy()

        if self.current_scenario_df is None or self.current_scenario_df.empty:
            self._clear_plotly_view(self.scenario_editor_canvas, "Données de base non disponibles.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.current_scenario_df['Date'], 
            y=self.current_scenario_df['VRP'], 
            mode='lines+markers', 
            name='VRP (éditable)'
        ))
        
        fig.update_layout(
            title="Éditeur de Scénario VRP (utilisez les outils 'lasso' ou 'box select' pour éditer)",
            template='plotly_dark',
            dragmode='lasso' 
        )
        
        fig.layout.template.data.scatter = [go.Scatter(selected=go.scatter.Selected(marker=dict(color='yellow')))]
        
        html = fig.to_html(
            include_plotlyjs='cdn', 
            config={'editable': True, 'edits': {'shapePosition': True}}
        )
        self.scenario_editor_canvas.setHtml(html)

        channel = QWebChannel(self.scenario_editor_canvas.page())
        self.scenario_editor_canvas.page().setWebChannel(channel)
        channel.registerObject("bridge", self.bridge)

        js_script = """
        <script src="qrcrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
            var bridge;
            new QWebChannel(qt.webChannelTransport, function (channel) {
                bridge = channel.objects.bridge;
            });

            function sendDataToPython() {
                var fig_div = document.getElementsByClassName('plotly-graph-div')[0];
                if (fig_div) {
                    var modified_y_data = fig_div.data[0].y;
                    bridge.receive_data(modified_y_data);
                }
            }
        </script>
        """
        html = fig.to_html(...) + js_script
        self.scenario_editor_canvas.setHtml(html)
    
    @pyqtSlot()
    def _trigger_scenario_analysis(self):
        """Demande au JS d'envoyer les données actuelles du graphique."""
        self.scenario_editor_canvas.page().runJavaScript("sendDataToPython();")

    @pyqtSlot(pd.DataFrame)
    def _handle_external_prediction_results(self, results_df):
        if 'Ramp' in results_df.columns and not results_df['Prediction'].isnull().all():
            metrics_df = results_df.dropna(subset=['Prediction', 'Ramp'])
            y_true = metrics_df['Ramp']
            y_pred = metrics_df['Prediction']
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            self.pred_report_text.setText(json.dumps(report, indent=2))
            
            class_names = sorted(list(set(y_true) | set(y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=class_names)
            fig_cm = ff.create_annotated_heatmap(z=cm, x=class_names, y=class_names, colorscale='Blues')
            fig_cm.update_layout(title="Matrice de Confusion", template='plotly_dark')
            self.pred_confusion_matrix_canvas.setHtml(fig_cm.to_html(include_plotlyjs='cdn'))
        else:
            self.pred_report_text.clear()
            self.pred_confusion_matrix_canvas.setHtml("")

        fig_plot = go.Figure()
        fig_plot.add_trace(go.Scatter(x=results_df['Date'], y=results_df['VRP'], mode='markers', name='VRP Data', marker=dict(size=3, color='grey')))
        
        preds_df = results_df.dropna(subset=['Prediction'])
        if not preds_df.empty:
            fig_plot.add_trace(go.Scatter(x=preds_df['Date'], y=preds_df['VRP'], mode='markers', name='Prédiction "Active"', marker=dict(size=8, color='orangered', symbol='x')))

        if 'Ramp' in results_df.columns:
            events_df = results_df[results_df['Ramp'] != 'Calm']
            event_starts = events_df[events_df['Ramp'].shift(1) != events_df['Ramp']]
            for date in event_starts['Date']:
                fig_plot.add_vline(x=date, line_width=2, line_dash="dash", line_color="red")
        
        fig_plot.update_layout(title="Performance sur Données Externes", yaxis_title="VRP [MW]", template='plotly_dark')
        self.pred_results_canvas.setHtml(fig_plot.to_html(include_plotlyjs='cdn'))

    @pyqtSlot(int)
    def _update_predictions_with_threshold(self, slider_value):
        from sklearn.metrics import precision_score, recall_score, classification_report

        if not self.last_training_results_for_tuning:
            return

        threshold = slider_value / 100.0
        self.threshold_value_label.setText(f"Seuil : {threshold:.2f}")

        results = self.last_training_results_for_tuning
        probabilities = results['probabilities']
        y_test = results['plot_data']['y_test']
        class_names = results['class_names']

        positive_class_name = next((c for c in class_names if c != 'Calm'), 'Actif')
        positive_class_index = class_names.index(positive_class_name)
        prob_positive = probabilities[:, positive_class_index]
        
        final_predictions = np.where(prob_positive >= threshold, positive_class_name, 'Calm')
        
        for label_name, (prob_min, prob_max) in self.ephemeral_label_rules.items():
            mask = (prob_positive >= prob_min) & (prob_positive < prob_max)
            final_predictions[mask] = label_name
        
        y_test_binary = np.where(y_test == 'Calm', 'Calm', 'Actif')
        y_pred_binary_for_metrics = np.where(final_predictions == 'Calm', 'Calm', 'Actif')
        
        precision = precision_score(y_test_binary, y_pred_binary_for_metrics, pos_label='Actif', zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary_for_metrics, pos_label='Actif', zero_division=0)
        self.threshold_metrics_label.setText(f"Précision ('Actif'): {precision:.2f} | Rappel ('Actif'): {recall:.2f}")
        
        base_report = classification_report(y_test, np.where(prob_positive >= threshold, positive_class_name, 'Calm'), output_dict=True, zero_division=0)
        self.update_metrics_plot({'report': base_report})
        self.update_main_plot(y_test, final_predictions)
        
    @pyqtSlot(dict)
    def _handle_simulation_start(self, init_data):
        self.sim_start_button.setEnabled(False)
        self.sim_pause_button.setEnabled(True); self.sim_pause_button.setText("PAUSE")
        self.sim_stop_button.setEnabled(True)
        self.sim_model_combo.setEnabled(False)

        self.sim_all_dates = pd.to_datetime(init_data['dates'])
        self.sim_all_vrp = np.array(init_data['vrp'])
        self.sim_all_truth = np.array(init_data['true_labels'])
        self.sim_model_classes = init_data['model_classes']

        self.sim_predictions = np.empty(len(self.sim_all_dates), dtype=object)
        self.sim_probabilities = np.zeros((len(self.sim_all_dates), len(self.sim_model_classes)))

        self.simulation_vrp_canvas.setHtml("")
        self.simulation_prob_canvas.setHtml("")

    @pyqtSlot(dict)
    def update_simulation_plots(self, step_data):
        idx = step_data['index']
        
        self.sim_predictions[idx] = step_data['prediction']
        self.sim_probabilities[idx, :] = step_data['probabilities']
        
        view_window = 200
        start_idx = max(0, idx - view_window)
        end_idx = idx + 1
        
        dates_view = self.sim_all_dates[start_idx:end_idx]
        
        fig_vrp = go.Figure()
        fig_vrp.add_trace(go.Scatter(x=dates_view, y=self.sim_all_vrp[start_idx:end_idx], mode='lines+markers', name='VRP', line=dict(color='grey')))
        
        label_map = {'Calm': 0, 'Pre-Event': 1, 'High-Paroxysm': 2, 'Post-Event': -1, 'Actif': 1, 'Low':0}
        
        true_labels_numeric = [label_map.get(l, -2) for l in self.sim_all_truth[start_idx:end_idx]]
        fig_vrp.add_trace(go.Scatter(x=dates_view, y=true_labels_numeric, mode='markers', name='Vraie Nature', marker=dict(color='deepskyblue', symbol='circle-open')))
        
        pred_labels_numeric = [label_map.get(l, -2) for l in self.sim_predictions[start_idx:end_idx] if l is not None]
        pred_dates = dates_view[:len(pred_labels_numeric)]
        fig_vrp.add_trace(go.Scatter(x=pred_dates, y=pred_labels_numeric, mode='markers', name='Prédiction', marker=dict(color='red', symbol='x', size=10)))
        
        fig_vrp.update_layout(title=f"Simulation en Temps Réel (Point {idx})", template='plotly_dark', yaxis_title="VRP / État")
        self.simulation_vrp_canvas.setHtml(fig_vrp.to_html(include_plotlyjs='cdn'))
        
        fig_prob = go.Figure()
        for i, class_name in enumerate(self.sim_model_classes):
            fig_prob.add_trace(go.Scatter(x=dates_view, y=self.sim_probabilities[start_idx:end_idx, i], mode='lines', name=f'P({class_name})'))
        
        fig_prob.update_layout(title="Probabilités en Temps Réel", template='plotly_dark', yaxis_title="Probabilité", yaxis_range=[0,1])
        self.simulation_prob_canvas.setHtml(fig_prob.to_html(include_plotlyjs='cdn'))

    @pyqtSlot(str)
    def _handle_simulation_finish(self, message):
        self.sim_start_button.setEnabled(True)
        self.sim_pause_button.setEnabled(False); self.sim_pause_button.setText("PAUSE")
        self.sim_stop_button.setEnabled(False)
        self.sim_model_combo.setEnabled(True)
        QMessageBox.information(self, "Simulation Terminée", message)

    @pyqtSlot(bool)
    def _toggle_all_features(self, checked):
        """Coche ou décoche tous les items de la liste de caractéristiques."""
        for i in range(self.feature_list_widget.count()):
            item = self.feature_list_widget.item(i)
            item.setSelected(checked)

    @pyqtSlot()
    def _restore_default_features(self):
        """Restaure la liste de caractéristiques à son état d'origine."""
        self.feature_list_widget.clear()
        self.feature_list_widget.addItems(self.default_features)
        self.select_all_cb.setChecked(True)
        self._toggle_all_features(True)
        self._show_message_box("Succès", "La liste complète des caractéristiques a été restaurée.")

    @pyqtSlot()
    def _select_top_n_features(self):
        """Sélectionne les N caractéristiques les plus importantes du dernier entraînement."""
        if not self.controller.last_training_results or 'model' not in self.controller.last_training_results:
            self._show_message_box("Avertissement", 
                "Cette fonction nécessite qu'un modèle ait été entraîné au moins une fois.\n"
                "Veuillez lancer un entraînement dans l'onglet 3.")
            return

        model = self.controller.last_training_results['model']
        if not hasattr(model, 'feature_importances_'):
            self._show_message_box("Info", "Le modèle actuel ne fournit pas d'importance de caractéristiques (ex: KNN).")
            return

        importances = model.feature_importances_
        names = model.feature_names_in_
        
        feature_importance_df = pd.DataFrame({'feature': names, 'importance': importances})
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

        n_to_select = self.top_n_spin.value()
        top_n_features = feature_importance_df['feature'].head(n_to_select).tolist()

        self._toggle_all_features(False) 
        for i in range(self.feature_list_widget.count()):
            item = self.feature_list_widget.item(i)
            if item.text() in top_n_features:
                item.setSelected(True)
                
        self.select_all_cb.setChecked(False)
        self._show_message_box("Succès", f"Les {len(top_n_features)} caractéristiques les plus importantes ont été sélectionnées.")
    def _connect_signals(self):
        # Onglet 1: Gestion Données
        self.load_button.clicked.connect(self._trigger_load_data)
        self.btn_bal.clicked.connect(self.controller.run_balancing_preview)
        
        # Onglet 2: Extraction Caractéristiques
        self.gen_btn.clicked.connect(self._trigger_feature_extraction)
        self.feature_list_widget.currentItemChanged.connect(self.update_feature_plots)

    # --- NOUVELLES CONNEXIONS ---
        self.select_all_cb.toggled.connect(self._toggle_all_features)
        self.restore_features_btn.clicked.connect(self._restore_default_features)
        self.select_top_n_btn.clicked.connect(self._select_top_n_features)
        # Dans la méthode _connect_signals de MainWindow

        self.select_nn_preset_btn.clicked.connect(self._select_nn_preset_features)

        # Onglet 3: Entraînement
        self.run_training_button.clicked.connect(self._trigger_run_training)
        self.clear_button.clicked.connect(self.clear_results)
        self.save_model_button.clicked.connect(self._trigger_save_model)
        self.model_type_combo.currentIndexChanged.connect(self.params_stack.setCurrentIndex)
        self.threshold_slider.valueChanged.connect(self._update_predictions_with_threshold)
        self.add_ephemeral_label_btn.clicked.connect(self._add_or_update_ephemeral_label)
        self.remove_ephemeral_label_btn.clicked.connect(self._remove_ephemeral_label)

        # Onglet 4: Comparaison
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.compare_button.clicked.connect(self._trigger_comparison)
        self.leaderboard_button.clicked.connect(self.controller.show_leaderboard)
        
        # Onglet 5: Prédiction Externe
        self.pred_load_data_button.clicked.connect(self._trigger_load_external_data)
        self.pred_run_button.clicked.connect(self._trigger_run_external_prediction)

        # Signaux du Contrôleur vers la Vue
        self.controller.status_updated.connect(self.update_status_label)
        self.controller.error_occurred.connect(self._show_message_box)
        self.controller.data_loaded.connect(self.display_dataframe)
        self.controller.overview_ready.connect(self.update_overview_display)
        self.controller.feature_extraction_finished.connect(self._handle_feature_extraction_results)
        self.controller.training_finished.connect(self._handle_training_results)
        self.controller.model_list_updated.connect(self._update_model_combos)
        self.controller.comparison_ready.connect(self._display_comparison)
        self.controller.leaderboard_ready.connect(self._display_leaderboard)
        self.controller.external_data_loaded.connect(self.update_status_label)
        self.controller.external_prediction_finished.connect(self._handle_external_prediction_results)

        self.sim_start_button.clicked.connect(self._trigger_start_simulation)
        self.sim_pause_button.clicked.connect(self.controller.pause_resume_simulation)
        self.sim_stop_button.clicked.connect(self.controller.stop_simulation)

        # Connexions du contrôleur vers la vue pour la simulation
        self.controller.simulation_started.connect(self._handle_simulation_start)
        self.controller.simulation_step_updated.connect(self.update_simulation_plots)
        self.controller.simulation_finished.connect(self._handle_simulation_finish)

        self.explain_button.clicked.connect(self._trigger_explanation)
        self.controller.explanation_ready.connect(self._display_shap_explanation)
        self.tabs.currentChanged.connect(self._on_xai_tab_selected) 

        self.feature_selector_combo.currentIndexChanged.connect(self.update_dependence_plot)
        self.interaction_selector_combo.currentIndexChanged.connect(self.update_dependence_plot)

        #Connexion avec le segmentateur

        self.train_segmenter_button.clicked.connect(self._trigger_train_segmenter)
        self.controller.segmentation_log_updated.connect(self.seg_log_text.append)
        self.controller.segmentation_finished.connect(self._handle_segmentation_results)

        self.controller.whatif_analysis_finished.connect(self._display_scenario_result)

    
    @pyqtSlot(dict)
    def _display_scenario_result(self, results):
        probabilities = results['probabilities']
        dates = results['dates']
        class_names = results['class_names']
        
        prob_df = pd.DataFrame(probabilities, index=dates, columns=class_names)
        
        fig = make_subplots(rows=len(class_names), cols=1, shared_xaxes=True, 
                            subplot_titles=[f"Probabilité Prédite de '{name}'" for name in class_names])
        
        for i, class_name in enumerate(class_names):
            fig.add_trace(go.Scatter(x=prob_df.index, y=prob_df[class_name], name=f'P({class_name})'), row=i+1, col=1)

        fig.update_layout(title="Résultat de l'Analyse de Scénario", template='plotly_dark', showlegend=False)
        self._display_plotly_fig(self.scenario_result_canvas, fig)

    def _trigger_load_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Sélectionner un Fichier de Données", "./data", "Fichiers Excel (*.xlsx)")
        if not filepath:
            return

        # Le worker s'exécute en arrière-plan pour charger les données
        worker = self.controller.load_data_from_file(filepath)

        if worker:
            image_path = "Core/BAHAHAHA/NINHO.png"

            # Créer le dialogue d'image
            dialog = ImageLoadingDialog(image_path, self)

            # La seule chose que nous attendons, c'est que le worker finisse.
            # Quand il a fini, il émet un signal qui ferme notre dialogue.
            worker.finished.connect(dialog.close)
            
            # Il bloquera l'interface principale (ce qui est voulu)
            # et se fermera automatiquement quand le worker sera terminé.
            dialog.exec()
    
    def _trigger_feature_extraction(self):
        config = {
            'features': [item.text() for item in self.feature_list_widget.selectedItems()],
            'window_sizes': {'win10': self.win10_spin.value(), 'win30': self.win30_spin.value(), 'win50': self.win50_spin.value()},
        }
        
        # Lancer la tâche et récupérer le worker
        worker = self.controller.run_feature_extraction(config)

        if worker:
            dialog = ProcessingDialog("Extraction des Caractéristiques", self)
            # Connexions communes
            worker.progress_updated.connect(dialog.update_progress)
            dialog.cancelled.connect(worker.cancel)
            worker.finished.connect(dialog.task_finished)
            dialog.exec()

    def _trigger_run_training(self):
        """
        Collecte tous les paramètres de l'interface, les assemble dans un dictionnaire
        et lance la tâche d'entraînement via le contrôleur.
        """
        self.clear_results()
        
        
        params = {
            # Vient de la liste déroulante des modèles
            'model_type': self.model_type_combo.currentText(),
            
            # Viennent des options de partitionnement
            'split_type': self.split_type_combo.currentText(),
            'train_ratio': self.train_ratio_spin.value(),
            'val_ratio': self.val_ratio_spin.value(),
            'test_ratio': self.test_ratio_spin.value(),
            
            # Viennent des checkboxes de définition de la cible
            'use_binary_mode': self.rb_binary_mode.isChecked(),
            'use_cycle_target': self.rb_multiclass_mode.isChecked(),
            'pre_event_ratio': self.pre_event_ratio_spin.value(),
            'post_event_ratio': self.post_event_ratio_spin.value(),
            
            # Viennent des options générales et avancées
            'use_balancing': self.cb_bal.isChecked(),
            'use_sample_weight': self.cb_sample_weight.isChecked(),
            'decay_rate': self.decay_rate_spin.value(),
            'normalize': self.norm_cb.isChecked(),
            
            # Vient de l'onglet d'extraction de caractéristiques
            'feature_config': {
                'features': [item.text() for item in self.feature_list_widget.selectedItems()],
                'window_sizes': {
                    'win10': self.win10_spin.value(), 
                    'win30': self.win30_spin.value(), 
                    'win50': self.win50_spin.value()
                },
            },
            
            # Vient du "StackedWidget" des hyperparamètres du modèle
            'model_params': {
                'max_k': self.max_k_spin.value(),
                'weights': [w.text() for w in [self.weights_uniform_cb, self.weights_distance_cb] if w.isChecked()],
                'metrics': [m.text() for m in [self.metric_euclidean_cb, self.metric_manhattan_cb] if m.isChecked()],
                'max_trees': self.rf_trees_spin.value(),
                'max_depth': self.rf_max_depth_spin.value(),
                'min_leaf': self.rf_min_leaf_spin.value(),
                'use_class_weight': self.rf_class_weight_cb.isChecked(),

                'nn_epochs': self.nn_epochs_spin.value(),
                'nn_batch_size': self.nn_batch_size_spin.value(),
                'nn_learning_rate': self.nn_lr_spin.value()
        
            }
        }
        
        # L'appel au contrôleur, en lui passant le dictionnaire complet
        worker = self.controller.run_full_training(params)

        # La suite gère l'affichage de la fenêtre de progression
        if worker:
            dialog = ProcessingDialog("Entraînement du Modèle", self)
            worker.progress_updated.connect(dialog.update_progress)
            dialog.cancelled.connect(worker.cancel)
            worker.finished.connect(dialog.task_finished)
            dialog.exec()
        
        # Si worker est None, la fonction se termine silencieusement.
        # Le contrôleur aura déjà affiché une popup d'erreur grâce au décorateur.
    

    def _trigger_save_model(self):
        model_name = self.model_name_input.text().strip().replace(" ", "_")
        self.controller.save_current_model(model_name)
    
    def _on_tab_changed(self, index):
        if self.tabs.tabText(index) in ["4. Comparaison", "5. Prédiction Externe", "6. Simulation Temps Réel"]:
            self.controller.refresh_model_lists()
            
    def _trigger_comparison(self):
        model_a = self.model_a_combo.currentText(); model_b = self.model_b_combo.currentText()
        self.controller.run_comparison(model_a, model_b)
        
    def _trigger_load_external_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Charger Données Externes", "./data", "Fichiers Excel (*.xlsx)")
        if not filepath:
            return

        dialog = ProcessingDialog("Chargement des Données Externes", self)
        worker = self.controller.load_external_data(filepath)

        # Connexions communes
        worker.progress_updated.connect(dialog.update_progress)
        dialog.cancelled.connect(worker.cancel)
        worker.finished.connect(dialog.task_finished)

        dialog.exec()
            

    def _trigger_run_external_prediction(self):
        model_name = self.pred_model_combo.currentText()
        replicate_test_set = self.pred_replicate_group.isChecked()
        test_set_percentage = self.pred_percentage_spin.value()
        
        # L'appel au contrôleur inclut maintenant les nouveaux paramètres
        worker = self.controller.run_external_prediction(
            model_name, 
            replicate_test_set, 
            test_set_percentage
        )

        if worker:
            dialog = ProcessingDialog("Prédiction sur Données Externes", self)
            worker.progress_updated.connect(dialog.update_progress)
            dialog.cancelled.connect(worker.cancel)
            worker.finished.connect(dialog.task_finished)
            dialog.exec()
    def update_learning_curves_plot(self, results):
        history = results.get('training_history')
        if not history: return

        epochs = list(range(1, len(history['loss']) + 1))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=epochs, y=history['loss'], name='Perte (Entraînement)'), secondary_y=False)
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Perte (Validation)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], name='Précision (Entraînement)'), secondary_y=True)
        if 'val_accuracy' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Précision (Validation)'), secondary_y=True)
        fig.update_layout(title="Courbes d'Apprentissage", template='plotly_dark')
        fig.update_yaxes(title_text="Perte (Loss)", secondary_y=False)
        fig.update_yaxes(title_text="Précision (Accuracy)", secondary_y=True)
        self._display_plotly_fig(self.learning_curves_canvas, fig)

    @pyqtSlot(str)
    def update_status_label(self, text):
        self.status_label.setText(text)
        
    @pyqtSlot(str, str)
    def _show_message_box(self, title, message):
        if title.lower() == "erreur": QMessageBox.critical(self, title, message)
        elif title.lower() == "succès": QMessageBox.information(self, title, message)
        else: QMessageBox.warning(self, title, message)
            
    @pyqtSlot(pd.DataFrame)
    def display_dataframe(self, df):
        # Cette fonction affiche maintenant UNIQUEMENT le DataFrame reçu dans la table.
        # Elle ne tente plus de synchroniser l'état de l'application.
        model = PandasModel(df)
        self.table_view.setModel(model)

    @pyqtSlot(pd.DataFrame)
    def update_overview_display(self, df):
        self.original_df = df.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['VRP'], mode='lines', name='VRP', line=dict(color='grey')))
        
        if 'Ramp' in df.columns:
            high_data = df[df['Ramp'] == 'High']
            if not high_data.empty:
                fig.add_trace(go.Scatter(x=high_data['Date'], y=high_data['VRP'], mode='markers', name='"High" Event', marker=dict(color='orangered', size=5, opacity=0.7)))

        fig.update_layout(
            title="Times Series Overview",
            yaxis_title="VRP [MW]",
            template='plotly_dark'
        )
        self._display_plotly_fig(self.overview_canvas, fig)
        
        stats_text = ""; desc = df['VRP'].describe()
        stats_text += f"Points totaux: {int(desc['count'])}\n"; stats_text += f"Moyenne: {desc['mean']:.2f}\n"
        stats_text += f"Écart-type: {desc['std']:.2f}\n"; stats_text += f"Min / Max: {desc['min']:.2f} / {desc['max']:.2f}\n\n"
        if 'Ramp' in df.columns:
            label_counts = df['Ramp'].value_counts(); stats_text += "Distribution des Labels Actuels:\n"
            for label, count in label_counts.items(): stats_text += f" - {label}: {count} points\n"
        self.stats_text.setText(stats_text)

    @pyqtSlot(pd.DataFrame)
    def _handle_feature_extraction_results(self, feature_matrix):
        """
        CORRIGÉ : Ne modifie plus la liste des caractéristiques de l'UI.
        Met simplement à jour les données pour les graphiques d'analyse.
        """
        self.feature_matrix_for_analysis = feature_matrix
        
        # Met à jour les graphiques et sélectionne le premier item pour l'affichage
        self.update_correlation_plot()
        if self.feature_list_widget.count() > 0:
            # Sélectionne le premier item coché pour afficher ses plots
            selected_items = self.feature_list_widget.selectedItems()
            if selected_items:
                self.feature_list_widget.setCurrentItem(selected_items[0])
            else:
                self.feature_list_widget.setCurrentRow(0)
            

    @pyqtSlot(dict)
    def _handle_training_results(self, results):
        # Stocker les résultats pour le tuning interactif
        self.last_training_results_for_tuning = results
        self.threshold_slider.setEnabled(True)

        plot_data = results.get('plot_data', {})
        self.train_df_for_plot = plot_data.get('train_df')
        self.val_df_for_plot = plot_data.get('val_df')
        self.test_df_for_plot = plot_data.get('test_df')
        
        self.update_metrics_display(results)
        self.update_metrics_plot(results) 
        self.update_confusion_matrix(results['confusion_matrix'], results['class_names'])
        self.update_feature_importance_plot(results)
        self.update_permutation_importance_plot(results)
        self.update_learning_curves_plot(results)
        self.update_probability_evolution_plot(results)

        model_name = self.model_name_input.text()

        # On met à jour le label dans l'onglet "Scénarios (What-If)"
        if model_name:
            self.scenario_model_label.setText(model_name)
            # On active le bouton d'analyse car un modèle est maintenant disponible
            self.analyze_scenario_btn.setEnabled(True)
        else:
            # Sécurité si aucun nom n'a été généré
            self.scenario_model_label.setText("Modèle non nommé")
            self.analyze_scenario_btn.setEnabled(False)
        
        # On rafraîchit le graphique de l'éditeur de scénario avec les nouvelles données de test/validation
        self._setup_scenario_editor()
        
        # Mettre à jour avec le seuil initial (0.50)
        self._update_predictions_with_threshold(50) 
        self.threshold_slider.setValue(50)
        
        self.save_model_button.setEnabled(True)
        
        model_type = self.model_type_combo.currentText()
        if "Random Forest" in model_type: model_tag = "RF"
        elif "LightGBM" in model_type: model_tag = "LGBM"
        else: model_tag = "KNN"

        split_tag = "Time" if self.split_type_combo.currentText().startswith("Temporel") else "Rand"
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
        best_p = results['best_params']
        
        if self.controller.last_feature_config:
            num_features = len(self.controller.last_feature_config['features'])
        else:
            num_features = 'N/A'
        
        if model_tag in ["RF", "LGBM"]:
            params = best_p.get('model', {}).get_params() if 'model' in best_p else best_p
            n_estimators = params.get('n_estimators', 'N/A')
            max_depth = params.get('max_depth', 'N/A')
            suggested_name = f"{model_tag}-F{num_features}-{split_tag}-t{n_estimators}d{max_depth}-id{timestamp}"
        else:
            weight_tag = "dist" if best_p.get('weights') == 'distance' else "uni"
            k_neighbors = best_p.get('n_neighbors', 'N/A')
            suggested_name = f"KNN-F{num_features}-{split_tag}-{weight_tag}_k{k_neighbors}-id{timestamp}"
            
        self.model_name_input.setText(suggested_name)

    @pyqtSlot(list)
    def _update_model_combos(self, models):
        current_tab_text = self.tabs.tabText(self.tabs.currentIndex())
        combos_to_update = []
        
        # Déterminer quels combos mettre à jour en fonction de l'onglet actif
        if current_tab_text == "4. Comparaison":
            combos_to_update = [self.model_a_combo, self.model_b_combo]
        elif current_tab_text == "5. Prédiction Externe":
            combos_to_update = [self.pred_model_combo]
        elif current_tab_text == "6. Simulation Temps Réel": 
            combos_to_update = [self.sim_model_combo]

        # Logique générique pour remplir les combos
        for combo in combos_to_update:
            current_selection = combo.currentText()
            combo.clear()
            combo.addItems(models)
            if current_selection in models:
                combo.setCurrentText(current_selection)

    @pyqtSlot(dict, dict)
    def _display_comparison(self, report_a, report_b):
        self.comparison_tabs.setCurrentWidget(self.avsb_tab)
        model_a_name, model_b_name = report_a['model_name'], report_b['model_name']
        
        all_labels = sorted(list(set(report_a.get('report', {}).keys()) | set(report_b.get('report', {}).keys()) - {'accuracy', 'macro avg', 'weighted avg'}))
        metrics = ["f1-score", "precision", "recall"]
        
        self.comparison_table.clear()
        self.comparison_table.setColumnCount(2)
        self.comparison_table.setHorizontalHeaderLabels([model_a_name, model_b_name])
        
        row_count = 0
        # On calcule le nombre de lignes nécessaires
        num_rows = (len(all_labels) + 2) * len(metrics)
        self.comparison_table.setRowCount(num_rows)
        
        for group_name in ['macro avg', 'weighted avg'] + all_labels:
            for metric in metrics:
                header = f"{group_name.replace(' avg', '').title()} {metric.replace('-',' ').title()}"
                self.comparison_table.setVerticalHeaderItem(row_count, QTableWidgetItem(header))
                
                # Accès sécurisé aux valeurs
                val_a = report_a.get('report', {}).get(group_name, {}).get(metric, 0)
                val_b = report_b.get('report', {}).get(group_name, {}).get(metric, 0)
                
                self.comparison_table.setItem(row_count, 0, QTableWidgetItem(f"{val_a:.4f}"))
                self.comparison_table.setItem(row_count, 1, QTableWidgetItem(f"{val_b:.4f}"))
                row_count += 1
                
        self.comparison_table.resizeColumnsToContents()

        radar_labels = [lbl.title() for lbl in all_labels]
        
        fig = go.Figure()
        
        stats_a = [report_a.get('report', {}).get(lbl, {}).get('f1-score', 0) for lbl in all_labels]
        stats_b = [report_b.get('report', {}).get(lbl, {}).get('f1-score', 0) for lbl in all_labels]
        
        fig.add_trace(go.Scatterpolar(
            r=stats_a,
            theta=radar_labels,
            fill='toself',
            name=model_a_name
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=stats_b,
            theta=radar_labels,
            fill='toself',
            name=model_b_name
        ))
        
        # Mise en forme du graphique
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  
                )
            ),
            showlegend=True,
            title="Comparaison Radar des F1-Scores par Classe",
            template='plotly_dark',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        # Affichage dans le QWebEngineView
        self.radar_canvas.setHtml(fig.to_html(include_plotlyjs='cdn'))
        
    @pyqtSlot(list)
    def _display_leaderboard(self, sorted_reports):
        self.comparison_tabs.setCurrentWidget(self.leaderboard_tab)
        headers = ["Rang", "Nom", "F1 Macro", "Précision", "Rappel", "Meilleurs Params"]
        self.leaderboard_table.setColumnCount(len(headers)); self.leaderboard_table.setHorizontalHeaderLabels(headers)
        self.leaderboard_table.setRowCount(len(sorted_reports))
        for row, report in enumerate(sorted_reports):
            self.leaderboard_table.setItem(row, 0, QTableWidgetItem(f"#{row+1}"))
            self.leaderboard_table.setItem(row, 1, QTableWidgetItem(report['model_name']))
            self.leaderboard_table.setItem(row, 2, QTableWidgetItem(f"{report['report']['macro avg']['f1-score']:.4f}"))
            self.leaderboard_table.setItem(row, 3, QTableWidgetItem(f"{report['report']['macro avg']['precision']:.4f}"))
            self.leaderboard_table.setItem(row, 4, QTableWidgetItem(f"{report['report']['macro avg']['recall']:.4f}"))
            self.leaderboard_table.setItem(row, 5, QTableWidgetItem(str(report['best_params'])))
        self.leaderboard_table.resizeColumnsToContents()


    @pyqtSlot(int)
    def _on_scenario_tab_selected(self, index):
        if self.tabs.tabText(index) == "Scénarios (What-If)":
            self._setup_scenario_editor()

    @pyqtSlot()
    def _trigger_train_segmenter(self):
        """Déclenche la pipeline d'entraînement du Segmenteur."""
        self.seg_log_text.clear()
        self.seg_report_text.clear()
        self.seg_confusion_matrix_canvas.axes.clear()
        self.seg_plot_canvas.axes.clear()

        # On peut passer les mêmes paramètres que l'entraînement principal pour l'instant
        params = {'model_params': {'max_trees': 150, 'max_depth': 10}}
        
        worker = self.controller.run_segmentation_training(params)

        if worker:
            dialog = ProcessingDialog("Entraînement du Segmenteur", self)
            worker.progress_updated.connect(dialog.update_progress)
            dialog.cancelled.connect(worker.cancel)
            worker.finished.connect(dialog.task_finished)
            dialog.exec()


    # Dans main2.py, REMPLACEZ cette fonction

    @pyqtSlot(dict)
    def _handle_segmentation_results(self, results):
        # Affichage du Rapport et de la Matrice de Confusion
        self.seg_report_text.clear()
        report = results.get('report', {})
        self.seg_report_text.setText(json.dumps(report, indent=2))
        
        cm = results.get('confusion_matrix')
        class_names = results.get('class_names')
        if cm is not None and len(cm) > 0 and class_names:
            fig_cm = ff.create_annotated_heatmap(z=cm, x=class_names, y=class_names, colorscale='Blues')
            fig_cm.update_layout(title="Matrice de Confusion du Segmenteur", template='plotly_dark')
            self._display_plotly_fig(self.seg_confusion_matrix_canvas, fig_cm)
        else:
            self._clear_plotly_view(self.seg_confusion_matrix_canvas, "Matrice de confusion non disponible.")

        # Préparation des données pour le graphique temporel
        y_test = results.get('plot_data', {}).get('y_test')
        y_pred_seg = results.get('plot_data', {}).get('predictions')
        
        if y_test is None or y_test.empty:
            self._clear_plotly_view(self.seg_plot_canvas, "Pas de données de test pour la visualisation.")
            self.seg_details_table.setRowCount(0)
            return
        
        # Le DataFrame d'analyse pour la table et le plot
        df_analysis = pd.DataFrame({
            'VRP': self.original_df.set_index('Date').loc[y_test.index, 'VRP'],
            'Original_Label': y_test.values, # Utiliser y_test directement pour la "vraie" phase
            'Segmenter_Prediction': y_pred_seg
        }).reset_index()

        # Mise à jour du graphique temporel avec Plotly
        fig_plot = go.Figure()
        fig_plot.add_trace(go.Scatter(x=df_analysis['Date'], y=df_analysis['VRP'], mode='lines', name='VRP Signal', line=dict(color='gray')))
        
        color_map = {'Pre-Event': 'rgba(255, 215, 0, 0.3)', 'High-Paroxysm': 'rgba(255, 69, 0, 0.4)', 'Post-Event': 'rgba(135, 206, 250, 0.4)'}
        
        pred_blocks = (df_analysis['Segmenter_Prediction'].shift() != df_analysis['Segmenter_Prediction']).cumsum()
        for _, group in df_analysis.groupby(pred_blocks):
            if not group.empty:
                phase = group['Segmenter_Prediction'].iloc[0]
                color = color_map.get(phase)
                if color:
                    fig_plot.add_vrect(x0=group['Date'].iloc[0], x1=group['Date'].iloc[-1], fillcolor=color, layer="below", line_width=0, annotation_text=phase, annotation_position="top left")

        fig_plot.update_layout(title="Prédictions du Segmenteur Superposées au Signal VRP", yaxis_title="VRP [MW]", template='plotly_dark')
        self._display_plotly_fig(self.seg_plot_canvas, fig_plot)
        
        # --- Remplissage de la table (avec correction pour les labels) ---
        self.seg_details_table.clear()
        headers = ["Date", "VRP", "Vraie Phase", "Prédiction Segmenteur"]
        self.seg_details_table.setColumnCount(len(headers))
        self.seg_details_table.setHorizontalHeaderLabels(headers)
        self.seg_details_table.setRowCount(len(df_analysis))
        
        for row_idx, row_data in df_analysis.iterrows():
            self.seg_details_table.setItem(row_idx, 0, QTableWidgetItem(str(row_data['Date'].strftime('%Y-%m-%d %H:%M'))))
            self.seg_details_table.setItem(row_idx, 1, QTableWidgetItem(f"{row_data['VRP']:.2f}"))
            self.seg_details_table.setItem(row_idx, 2, QTableWidgetItem(str(row_data['Original_Label'])))
            self.seg_details_table.setItem(row_idx, 3, QTableWidgetItem(str(row_data['Segmenter_Prediction'])))

        self.seg_details_table.resizeColumnsToContents()
    def clear_results(self):
        self.metrics_text.clear()
        self._clear_plotly_view(self.main_plot_canvas, "En attente des résultats de l'entraînement.")
        self._clear_plotly_view(self.metrics_plot_canvas, "En attente des résultats.")
        self._clear_plotly_view(self.probability_evolution_canvas, "En attente des résultats de l'entraînement.")
        self._clear_plotly_view(self.confusion_matrix_canvas, "En attente des résultats de l'entraînement.")
        self._clear_plotly_view(self.feature_importance_canvas, "En attente des résultats de l'entraînement.")
        self._clear_plotly_view(self.permutation_importance_canvas, "En attente des résultats.")
        self._clear_plotly_view(self.learning_curves_canvas, "En attente des résultats de l'entraînement.")
        self.status_label.setText("Prêt.")
        self.save_model_button.setEnabled(False)

    def update_feature_plots(self, current_item, previous_item):
        if current_item is None or self.feature_matrix_for_analysis is None: return
        feature_name = current_item.text()
        if feature_name not in self.feature_matrix_for_analysis.columns: return
        
        feature_data = self.feature_matrix_for_analysis[feature_name]
        
        # Histogramme
        fig_hist = go.Figure(data=[go.Histogram(x=feature_data, name='Distribution')])
        fig_hist.add_trace(go.Scatter(x=feature_data, y=[0]*len(feature_data), mode='markers', name='Points', marker=dict(opacity=0))) # Hack for KDE
        fig_hist.update_layout(title=f"'{feature_name}' Distribution", template='plotly_dark')
        self._display_plotly_fig(self.feature_dist_canvas, fig_hist)

        # Boxplot
        if self.original_df is not None and 'Ramp' in self.original_df.columns:
            df_labels_indexed = self.original_df.set_index('Date')
            labels_aligned = df_labels_indexed['Ramp'].reindex(self.feature_matrix_for_analysis.index, method='nearest')
            plot_df = pd.DataFrame({'feature': feature_data.values, 'label': labels_aligned.values})
            plot_df.dropna(inplace=True)

            fig_box = go.Figure()
            for label in plot_df['label'].unique():
                fig_box.add_trace(go.Box(y=plot_df[plot_df['label']==label]['feature'], name=label))
            fig_box.update_layout(title=f"'{feature_name}' vs. label", template='plotly_dark')
            self._display_plotly_fig(self.feature_boxplot_canvas, fig_box)
        
    def update_correlation_plot(self):
        if self.feature_matrix_for_analysis is None: return
        
        corr = self.feature_matrix_for_analysis.corr()
        fig = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdBu_r', zmid=0)) # Rouge-Bleu centré sur 0
        fig.update_layout(title="Correlation Matrix", template='plotly_dark')
        self._display_plotly_fig(self.corr_matrix_canvas, fig)
    
    def update_metrics_display(self, results):
        text=f"--- Meilleurs Hyperparamètres ---\n{json.dumps(results['best_params'], indent=2)}\n\n--- Rapport de Classification ---\n"
        if 'best_threshold' in results:
            text += f"--- Seuil de Décision Optimal Trouvé ---\n"
            text += f"Le seuil qui maximise le F1-Score est : {results['best_threshold']:.2f}\n\n"
        for cn, m in results['report'].items():
            if isinstance(m, dict): text += f"\nClasse: {cn}\n  Précision: {m['precision']:.4f}\n  Rappel: {m['recall']:.4f}\n  F1-Score: {m['f1-score']:.4f}\n"
        text += f"\n--- Scores Globaux ---\nMacro Avg F1: {results['report']['macro avg']['f1-score']:.4f}\nWeighted Avg F1: {results['report']['weighted avg']['f1-score']:.4f}\n"
        self.metrics_text.setText(text)
    
    def update_confusion_matrix(self, cm, class_names):
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(z=cm, x=class_names, y=class_names, colorscale='Blues')
        fig.update_layout(title="Confusion Matrix", template='plotly_dark')
        self._display_plotly_fig(self.confusion_matrix_canvas, fig)


    def update_main_plot(self, y_test, y_pred):
        if y_test is None or y_pred is None:
            self._clear_plotly_view(self.main_plot_canvas, "Données de prédiction non disponibles.")
            return

        fig = go.Figure()
        
        if self.train_df_for_plot is not None:
            fig.add_trace(go.Scatter(x=self.train_df_for_plot['Date'], y=self.train_df_for_plot['VRP'], mode='markers', name='Training set', marker=dict(size=4, opacity=0.5)))
        if self.val_df_for_plot is not None:
            fig.add_trace(go.Scatter(x=self.val_df_for_plot['Date'], y=self.val_df_for_plot['VRP'], mode='markers', name='Validation set', marker=dict(size=4, opacity=0.5)))
        if self.test_df_for_plot is not None:
            fig.add_trace(go.Scatter(x=self.test_df_for_plot['Date'], y=self.test_df_for_plot['VRP'], mode='markers', name='Test set', marker=dict(size=4, opacity=0.6)))

        y_pred_series = pd.Series(y_pred, index=y_test.index)
        
        color_map = {
            'Calm': 'royalblue', 'Actif': 'orangered', 'Pre-Event': 'gold', 
            'High-Paroxysm': 'red', 'Post-Event': 'skyblue'
        }
        
        all_pred_labels = sorted(y_pred_series.unique())
        
        color_index = 0
        for label in all_pred_labels:
            if label not in color_map:
                color_map[label] = f'hsl({(color_index*70)%360}, 80%, 60%)'
                color_index += 1

        all_labels_sorted = sorted(list(set(y_test.unique()) | set(all_pred_labels)))
        label_map_numeric = {label: i for i, label in enumerate(all_labels_sorted)}
        y_test_numeric = y_test.map(label_map_numeric)
        fig.add_trace(go.Scatter(
            x=y_test.index, y=y_test_numeric, mode='markers', name='True (State)', 
            marker=dict(symbol='circle-open', size=8, color='grey', line=dict(width=1)), yaxis="y2"
        ))
        
        for label in all_pred_labels:
            if label != 'Calm':
                preds_subset = y_pred_series[y_pred_series == label]
                if not preds_subset.empty:
                    fig.add_trace(go.Scatter(
                        x=preds_subset.index, y=preds_subset.map(label_map_numeric), 
                        mode='markers', name=f'Predicted ({label})', 
                        marker=dict(symbol='x', size=8, color=color_map.get(label)), yaxis="y2"
                    ))

        fig.update_layout(
            title="Prediction vs ground truth on the dataset",
            xaxis_title="Date", yaxis_title="VRP [MW]",
            yaxis2=dict(
                title="États du Système", overlaying='y', side='right', 
                tickvals=list(label_map_numeric.values()), ticktext=list(label_map_numeric.keys())
            ),
            template='plotly_dark', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        self._display_plotly_fig(self.main_plot_canvas, fig)
        
    def update_external_plot(self, df):
        ax = self.pred_results_canvas.axes
        ax.clear()
        ax.plot(df['Date'], df['VRP'], '.-', color='gray', alpha=0.6, label='Données VRP')
        ax.set_ylabel("VRP [MW]", color='gray')
        ax.tick_params(axis='y', labelcolor='gray')

        ax2 = ax.twinx()
        positive_class_map = {'High': 1, 'Pre-Event': 1}
        df['pred_numeric'] = df['Prediction'].map(positive_class_map).fillna(0)
        
        # Par défaut, on considère que toutes les prédictions sont des Faux Positifs (rouges)
        ax2.plot(df.loc[df['pred_numeric'] == 1, 'Date'], df.loc[df['pred_numeric'] == 1, 'pred_numeric'], 'x', color='red', markersize=8, label='Fausse Alerte (FP)')
        
        # Si les vrais labels existent, on peut faire la distinction
        if 'Ramp' in df.columns:
            df['vrai_numeric'] = df['Ramp'].map(positive_class_map).fillna(0)
            
            # Vrais événements
            ax2.plot(df.loc[df['vrai_numeric'] == 1, 'Date'], df.loc[df['vrai_numeric'] == 1, 'vrai_numeric'], 'o', color='deepskyblue', alpha=0.5, label='Vrai Événement')
            
            # Vrais Positifs (on les trace par-dessus les rouges pour les rendre verts)
            vrais_positifs = df[(df['pred_numeric'] == 1) & (df['vrai_numeric'] == 1)]
            ax2.plot(vrais_positifs['Date'], vrais_positifs['pred_numeric'], 'x', color='lime', markersize=10, label='Alerte Correcte (TP)')

        ax2.set_ylabel("Labels (0=Négatif, 1=Positif)", color='lime')
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis='y', labelcolor='lime')
        ax2.set_ylim(-0.1, 1.1)
        
        # Unifier les légendes des deux axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Éviter les doublons de labels
        unique_labels = dict(zip(labels2, lines2))
        ax2.legend(lines + list(unique_labels.values()), labels + list(unique_labels.keys()), loc='best')

        ax.set_title("Résultats du Modèle sur Données Externes")
        self.pred_results_canvas.figure.tight_layout()
        self.pred_results_canvas.draw()


    @pyqtSlot(object, object, object, list)
    def _display_shap_explanation(self, shap_values, base_value, instance_data, class_names):
        """
        Affiche le graphique SHAP (waterfall) en le recréant avec Plotly.
        """
        try:
            pred_proba = self.controller.current_explainer.model.predict_proba(instance_data.to_frame().T)[0]
            info_text = f"<b>Prédiction pour la date : {instance_data.name}</b><br>Probabilités prédites :<br>"
            for i, name in enumerate(class_names):
                info_text += f" - {name}: {pred_proba[i]:.2%}<br>"

            class_index = np.argmax(pred_proba)
            predicted_class_name = class_names[class_index]
            info_text += f"<br><i>Explication pour la classe : <b>'{predicted_class_name}'</b></i>"
            
            # Extraire les bonnes valeurs SHAP
            shap_values_for_class = shap_values[class_index]
            base_value_for_class = base_value[class_index] if isinstance(base_value, (list, np.ndarray)) else base_value

            # --- PARTIE 2 : Recréer le graphique Waterfall avec Plotly ---
            
            # Trier les features par leur impact SHAP
            sorted_indices = np.argsort(np.abs(shap_values_for_class))
            sorted_shap = shap_values_for_class[sorted_indices]
            sorted_features = instance_data.index[sorted_indices]
            sorted_data_values = instance_data.values[sorted_indices]

            # Calculer les positions des barres
            cumulative = base_value_for_class
            y_positions = []
            bar_starts = []
            for val in sorted_shap:
                bar_starts.append(cumulative)
                y_positions.append(f"{len(bar_starts)}. {sorted_features[len(bar_starts)-1]} = {sorted_data_values[len(bar_starts)-1]:.2f}")
                cumulative += val
            
            # Créer la figure Plotly
            fig = go.Figure()

            # Ajouter les barres
            fig.add_trace(go.Waterfall(
                x = sorted_shap,
                y = y_positions,
                orientation = "h",
                measure = ["relative"] * len(sorted_shap),
                base = base_value_for_class,
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))

            # Ajouter la barre de base et la barre finale
            fig.add_trace(go.Waterfall(
                x = [base_value_for_class, cumulative],
                y = ["Valeur de Base E(f(x))", "Prédiction Finale f(x)"],
                orientation = "h",
                measure = ["absolute", "absolute"],
            ))

            fig.update_layout(
                title="Décomposition de la Prédiction (Waterfall Plot)",
                template='plotly_dark',
                waterfallgap = 0.3,
            )

            self.xai_info_label.setText(info_text)
            self._display_plotly_fig(self.xai_waterfall_canvas, fig)

        except Exception as e:
            error_message = f"Erreur lors de la génération du graphique SHAP:\n{e}"
            self._clear_plotly_view(self.xai_waterfall_canvas, error_message)
            if 'info_text' in locals():
                self.xai_info_label.setText(info_text + f"<br><span style='color: orange;'>{error_message}</span>")
            import traceback
            traceback.print_exc()

    # À la fin de la classe MainWindow dans main2.py

    @pyqtSlot()
    def _select_nn_preset_features(self):
        """Sélectionne un sous-ensemble de caractéristiques jugées robustes pour un NN."""
        nn_features = [
            # Énergie et Volatilité
            "median30", "std30", "iqr30",
            # Forme de la distribution
            "kurtosis30", "skewness30",
            # Tendance et Accélération
            "slope30", "slope30_accel",
            # Contexte long terme
            "zscore_120",
            # Ratios de volatilité
            "volatility_ratio_10_30", "volatility_ratio_30_50",
            # Caractéristiques d'état
            "time_in_active_state", "energy_in_active_state"
        ]
        
        self._show_message_box("Info", f"{len(nn_features)} caractéristiques recommandées pour l'approche NN ont été sélectionnées.")

        # D'abord, on désélectionne tout
        for i in range(self.feature_list_widget.count()):
            self.feature_list_widget.item(i).setSelected(False)
            
        # Ensuite, on sélectionne uniquement celles du preset
        for i in range(self.feature_list_widget.count()):
            item = self.feature_list_widget.item(i)
            if item.text() in nn_features:
                item.setSelected(True)
        
        # S'assurer que la checkbox "Tout sélectionner" est décochée
        self.select_all_cb.setChecked(False)

    def _on_xai_tab_selected(self, index):
        """
        Met à jour l'onglet XAI lorsqu'il est sélectionné.
        VERSION CORRIGÉE : S'assure que les données de la table correspondent
        exactement aux données utilisées par l'explainer.
        """
        if self.tabs.tabText(index) != "7. Explicabilité (XAI)":
            return

        if self.controller.current_explainer:
            results = self.controller.last_training_results

            # C'est la seule garantie d'être synchronisé.
            X_test_for_table = self.controller.current_explainer.X_test_for_shap
            
            # On récupère les vrais labels et les prédictions correspondantes
            y_test = results['plot_data']['y_test'].loc[X_test_for_table.index]
            y_pred = pd.Series(results['predictions'], index=results['plot_data']['y_test'].index).loc[X_test_for_table.index]

            display_df = pd.DataFrame({
                'Date': X_test_for_table.index,
                'Vraie Classe': y_test.values,
                'Classe Prédite': y_pred.values
            })
            
            self.xai_data_table.clearContents() # Plus efficace que clear()
            self.xai_data_table.setRowCount(len(display_df))
            self.xai_data_table.setColumnCount(len(display_df.columns))
            self.xai_data_table.setHorizontalHeaderLabels(display_df.columns)
            
            for i, row in display_df.iterrows():
                for j, col_name in enumerate(display_df.columns):
                    # Formater la date pour une meilleure lisibilité
                    value = row[col_name]
                    if isinstance(value, pd.Timestamp):
                        item_text = value.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        item_text = str(value)
                    self.xai_data_table.setItem(i, j, QTableWidgetItem(item_text))
            
            self.xai_data_table.resizeColumnsToContents()

            # Remplir les listes déroulantes (logique inchangée)
            features = X_test_for_table.columns.tolist()
            current_feature = self.feature_selector_combo.currentText()
            current_interaction = self.interaction_selector_combo.currentText()
            self.feature_selector_combo.blockSignals(True); self.interaction_selector_combo.blockSignals(True)
            self.feature_selector_combo.clear(); self.interaction_selector_combo.clear()
            self.feature_selector_combo.addItems(features)
            self.interaction_selector_combo.addItem("Auto"); self.interaction_selector_combo.addItems(features)
            if current_feature in features: self.feature_selector_combo.setCurrentText(current_feature)
            if current_interaction in ["Auto"] + features: self.interaction_selector_combo.setCurrentText(current_interaction)
            self.feature_selector_combo.blockSignals(False); self.interaction_selector_combo.blockSignals(False)
            
            self.update_dependence_plot()

        else:
            self.xai_data_table.clearContents()
            self.xai_data_table.setRowCount(0)
            self.feature_selector_combo.clear()
            self.interaction_selector_combo.clear()
            self._clear_plotly_view(self.xai_waterfall_canvas, "Aucun modèle compatible avec SHAP n'est entraîné.")
            self._clear_plotly_view(self.xai_dependence_canvas, "Veuillez entraîner un modèle (ex: Random Forest).")
            self.xai_info_label.setText("Aucun modèle n'est entraîné ou le modèle actuel (ex: KNN) n'est pas compatible avec SHAP.")

    # Fonction utilitairer vérifier 
    def check_shap_version():
        """
        Vérifie la version de SHAP et retourne des informations utiles
        """
        try:
            import shap
            version = shap.__version__
            print(f"Version SHAP détectée: {version}")
            
            # Vérifications de compatibilité
            major, minor = map(int, version.split('.')[:2])
            
            if major == 0 and minor < 20:
                print("ATTENTION: Version SHAP trop ancienne. Mise à jour recommandée.")
                return "old"
            elif major == 0 and minor >= 41:
                print("Version SHAP moderne détectée.")
                return "modern" 
            else:
                print("Version SHAP intermédiaire détectée.")
                return "intermediate"
                
        except Exception as e:
            print(f"Erreur lors de la vérification de la version SHAP: {e}")
            return "unknown"

    def _trigger_explanation(self):
        selected_items = self.xai_data_table.selectedItems()
        if not selected_items:
            self._show_message_box("Avertissement", "Veuillez sélectionner une ligne dans le tableau.")
            return
        
        selected_row_index = selected_items[0].row()
        self.controller.explain_prediction(selected_row_index)

    def _display_plotly_fig(self, canvas: QWebEngineView, fig: go.Figure):
        """
        Affiche une figure Plotly dans un QWebEngineView en utilisant le thème de publication
        et ajoute un menu contextuel pour l'exportation.
        """
        # Appliquer notre template de publication
        fig.update_layout(template='publication')

        # Convertit la figure en un simple div HTML
        raw_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Enveloppe ce div dans notre propre template HTML
        full_html = f"""
        <html>
            <head>
                <style>
                    body {{ 
                        background-color: white; /* Fond blanc pour correspondre au style papier */
                        margin: 0; 
                        padding: 0; 
                    }}
                </style>
            </head>
            <body>
                {raw_html}
            </body>
        </html>
        """
        canvas.setHtml(full_html)
        
        self._add_export_menu(canvas, fig)

    def _clear_plotly_view(self, canvas: QWebEngineView, message: str):
        """
        Affiche un message dans un QWebEngineView avec un fond clair.
        """
        html = f"""
        <html>
            <head>
                <style>
                    body {{ 
                        background-color: white; /* MODIFIÉ */
                        color: #555;            /* MODIFIÉ */
                        font-family: 'Times New Roman', serif; /* MODIFIÉ */
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        padding: 0;
                    }}
                </style>
            </head>
            <body>
                <div>{message}</div>
            </body>
        </html>
        """
        canvas.setHtml(html)

    def _add_export_menu(self, canvas: QWebEngineView, fig: go.Figure):
        """
        Ajoute un menu contextuel à un QWebEngineView pour exporter la figure Plotly.
        """
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def show_context_menu(position):
            ''' On va corriger l'erreur sur le QMenu ici '''
            menu=QMenu()

            save_svg_action = menu.addAction("Enregistrer en SVG...")
            save_pdf_action = menu.addAction("Enregistrer en PDF...")
            save_png_action = menu.addAction("Enregistrer en PNG (haute résolution)...")

            action = menu.exec(canvas.mapToGlobal(position))

            if action == save_svg_action:
                self._export_figure(fig, "SVG", "Fichiers Vectoriels SVG (*.svg)")
            elif action == save_pdf_action:
                self._export_figure(fig, "PDF", "Documents PDF (*.pdf)")
            elif action == save_png_action:
                self._export_figure(fig, "PNG", "Images PNG (*.png)")

        # Éviter les connexions multiples si la fonction est appelée plusieurs fois sur le même canvas
        try:
            canvas.customContextMenuRequested.disconnect()
        except TypeError:
            pass # Aucune connexion n'existait, c'est normal la première fois
        
        canvas.customContextMenuRequested.connect(show_context_menu)

    def _export_figure(self, fig: go.Figure, file_format: str, dialog_filter: str):
        """
        Ouvre une boîte de dialogue pour sauvegarder la figure dans le format spécifié.
        """
        # Proposer un nom de fichier par défaut basé sur le titre du graphique
        default_name = fig.layout.title.text if fig.layout.title.text else "figure"
        default_name = default_name.replace(" ", "_").lower()
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Enregistrer la Figure en {file_format}",
            f"{default_name}.{file_format.lower()}",
            dialog_filter
        )

        if filepath:
            try:
                if file_format == 'PNG':
                    # Pour PNG, on augmente la résolution
                    fig.write_image(filepath, scale=3) # scale=3 pour une résolution 3x
                else:
                    # Pour les formats vectoriels, pas besoin de scale
                    fig.write_image(filepath)
                self.update_status_label(f"Figure sauvegardée avec succès : {filepath}")
            except Exception as e:
                self._show_message_box("Erreur d'Exportation", f"Impossible de sauvegarder la figure : {e}")

