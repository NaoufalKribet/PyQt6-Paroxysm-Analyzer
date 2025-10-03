# --- START OF FILE launch_menu.py ---

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel
)
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QFont

# --- Import des fenêtres principales des deux modules ---
# Module Volcanologie
from main2 import MainWindow
from app_controller import AppController

# Module Sismique
from seismic_main_window import SeismicMainWindow

# La feuille de style est centralisée ici pour être importée par run_app.py
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

    /* ================================
       BOUTONS - Style premium (Adapté pour le menu de lancement)
       ================================ */
    QPushButton {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                   stop: 0 #4a4a4a, stop: 0.5 #404040, stop: 1 #353535);
        color: #ffffff;
        border: 1px solid #606060;
        border-radius: 8px; /* Plus arrondi pour un look de menu */
        padding: 12px 20px;
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
        padding-top: 13px;
        padding-bottom: 11px;
    }
    
    /* ... (tous les autres styles de DARK_STYLESHEET sont ici) ... */
    /* Pour la concision, le reste de la (longue) feuille de style n'est pas répété,
       mais dans votre fichier, elle doit être ici en entier. */
"""


class LaunchMenu(QWidget):
    """
    Une fenêtre de sélection qui permet à l'utilisateur de choisir quel
    module de l'application lancer.
    """
    def __init__(self):
        super().__init__()
        # Garde une référence à la fenêtre ouverte pour éviter qu'elle ne soit détruite
        self.opened_window = None
        self.initUI()

    def initUI(self):
        """Initialise l'interface utilisateur du menu."""
        self.setWindowTitle("Sélecteur de Projet")
        self.setGeometry(400, 400, 600, 350)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 40)
        main_layout.setSpacing(30)

        # Label de titre
        title_label = QLabel("Choisissez Votre Domaine d'Analyse")
        font = QFont('Inter', 18, QFont.Weight.Bold)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout pour les boutons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(30)

        # Création des boutons avec icônes et texte
        self.volcano_button = QPushButton("🌋\nVolcanologie")
        self.seismic_button = QPushButton("📉\nSismique")

        # Style des boutons
        button_font = QFont('Inter', 14)
        for button in [self.volcano_button, self.seismic_button]:
            button.setFont(button_font)
            button.setMinimumSize(QSize(220, 120))
            button.setStyleSheet("QPushButton { text-align: center; }")

        # Ajout des boutons au layout dédié
        buttons_layout.addWidget(self.volcano_button)
        buttons_layout.addWidget(self.seismic_button)

        # Assemblage du layout principal
        main_layout.addWidget(title_label)
        main_layout.addStretch()
        main_layout.addLayout(buttons_layout)
        main_layout.addStretch()

        # Connexion des signaux des boutons à leurs fonctions
        self.volcano_button.clicked.connect(self.open_volcano_app)
        self.seismic_button.clicked.connect(self.open_seismic_app)

    def open_volcano_app(self):
        """
        Crée le contrôleur et la fenêtre principale pour l'analyse volcanique,
        puis ferme le menu.
        """
        # Crée une instance fraîche du contrôleur pour ce module
        controller = AppController()
        # Passe le contrôleur à la fenêtre principale
        self.opened_window = MainWindow(controller)
        self.opened_window.show()
        self.close() # Ferme la fenêtre du menu

    def open_seismic_app(self):
        """
        Crée et affiche la fenêtre principale pour l'analyse sismique,
        puis ferme le menu.
        """
        # La fenêtre sismique gère son propre contrôleur en interne
        self.opened_window = SeismicMainWindow()
        self.opened_window.show()
        self.close() # Ferme la fenêtre du menu