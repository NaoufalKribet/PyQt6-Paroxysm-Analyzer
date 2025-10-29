from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QProgressBar, QLabel,
    QPushButton, QTextEdit, QDialogButtonBox
)

class ProcessingDialog(QDialog):
    """
    Une fenêtre de dialogue modale et non bloquante pour afficher la progression
    des opérations longues. Elle inclut une barre de progression, un message d'état,
    un journal détaillé et un bouton d'annulation.
    """
    cancelled = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setMinimumSize(500, 300)
        self.setModal(True) 
        
        self.status_label = QLabel("Initialisation...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11pt;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; background-color: #2b2b2b;")

        self.cancel_button = QPushButton("Annuler")
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Journal détaillé :"))
        layout.addWidget(self.log_area)
        
        button_box = QDialogButtonBox()
        button_box.addButton(self.cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(button_box)
        
        self.cancel_button.clicked.connect(self._on_cancel)

    def _on_cancel(self):
        """Gère le clic sur le bouton d'annulation."""
        self.add_log_message("--- ANNULATION DEMANDÉE PAR L'UTILISATEUR ---")
        self.cancel_button.setDisabled(True)
        self.cancel_button.setText("Annulation en cours...")
        self.cancelled.emit()


    def update_progress(self, value: int, status_message: str):
        """Met à jour la barre de progression ET le message d'état."""
        self.progress_bar.setValue(value)
        self.status_label.setText(status_message)
        self.add_log_message(f"[{value}%] {status_message}")

    def add_log_message(self, message: str):
        """Ajoute un message au journal détaillé."""
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def closeEvent(self, event):
        """Empêche la fermeture de la fenêtre avec la croix (seul Annuler fonctionne)."""
        event.ignore()

    def task_finished(self):
        """Méthode à appeler lorsque la tâche est terminée avec succès."""
        self.add_log_message("\n--- TÂCHE TERMINÉE AVEC SUCCÈS ---")
        self.progress_bar.setValue(100)
        self.status_label.setText("Terminé !")
        self.cancel_button.setText("Fermer")
        self.cancel_button.setDisabled(False)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import Qt

class ImageLoadingDialog(QDialog):
    """
    Un dialogue modal simple qui affiche une animation GIF pendant le chargement.
    """
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
    
        self.setModal(True)
        self.setWindowTitle("Chargement en cours...")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.label = QLabel(self)
        self.movie = QMovie(image_path)
        
        if not self.movie.isValid():
            print(f"Erreur: Impossible de charger le GIF depuis {image_path}")
            self.label.setText("Chargement...")
        else:
            self.label.setMovie(self.movie)
            self.setFixedSize(self.movie.frameRect().size())
            self.movie.start()

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        self.setLayout(layout)
