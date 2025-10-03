# --- START OF FILE run_app.py ---

import sys
from PyQt6.QtWidgets import QApplication
import matplotlib.pyplot as plt

# Importez le nouveau menu de lancement et la feuille de style partagée
from launch_menu import LaunchMenu, DARK_STYLESHEET

if __name__ == "__main__":
    # Initialise l'application Qt
    app = QApplication(sys.argv)
    
    # Applique le thème sombre de manière cohérente
    plt.style.use('dark_background')
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Crée et affiche le menu de lancement au lieu de la fenêtre principale
    launch_menu = LaunchMenu()
    launch_menu.show()
    
    # Démarre la boucle d'événements de l'application
    sys.exit(app.exec())