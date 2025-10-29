import sys
from PyQt6.QtWidgets import QApplication
import matplotlib.pyplot as plt

from launch_menu import LaunchMenu, DARK_STYLESHEET

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    plt.style.use('dark_background')
    app.setStyleSheet(DARK_STYLESHEET)
    
    launch_menu = LaunchMenu()
    launch_menu.show()
    

    sys.exit(app.exec())
