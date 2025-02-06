import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
from PIL import Image
import numpy as np
from image_retrieval_system import ImageRetrievalSystem

class ImageRetrievalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Retrieval System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Load the retrieval system
        self.retrieval_system = ImageRetrievalSystem.load_state('retrieval_system_state.pkl')
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create top section for query image
        top_section = QHBoxLayout()
        
        # Query image section
        query_section = QVBoxLayout()
        self.query_label = QLabel("Query Image")
        self.query_image = QLabel()
        self.query_image.setFixedSize(300, 300)
        self.query_image.setStyleSheet("border: 2px solid black")
        select_button = QPushButton("Select Image")
        select_button.clicked.connect(self.select_image)
        
        query_section.addWidget(self.query_label)
        query_section.addWidget(self.query_image)
        query_section.addWidget(select_button)
        
        top_section.addLayout(query_section)
        
        # Create grid for similar images
        self.results_grid = QGridLayout()
        self.similar_images = []
        for i in range(5):  # 5 similar images
            label = QLabel()
            label.setFixedSize(200, 200)
            label.setStyleSheet("border: 1px solid gray")
            self.similar_images.append(label)
            self.results_grid.addWidget(label, i // 3, i % 3)
        
        # Add layouts to main layout
        layout.addLayout(top_section)
        layout.addLayout(self.results_grid)
        
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            # Display selected image
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.query_image.setPixmap(scaled_pixmap)
            
            # Find image index in the dataset
            image_name = os.path.basename(file_name)
            try:
                query_idx = self.retrieval_system.image_info_df[
                    self.retrieval_system.image_info_df['image_id'] == image_name
                ].index[0]
                
                # Get similar images
                results = self.retrieval_system.query_similar_images(query_idx, top_k=5)
                
                # Display similar images
                self.display_similar_images(results['similar_images'])
                
            except IndexError:
                print("Image not found in database")
    
    def display_similar_images(self, similar_images):
        # Clear previous images
        for label in self.similar_images:
            label.clear()
        
        # Display new similar images
        for i, img_info in enumerate(similar_images):
            if i < len(self.similar_images):
                # Construct image path - modify this according to your directory structure
                image_path = os.path.join('path_to_your_images', img_info['image_id'])
                if os.path.exists(image_path):
                    pixmap = QPixmap(image_path)
                    scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
                    self.similar_images[i].setPixmap(scaled_pixmap)
                    
                    # Add score label
                    score_label = QLabel(f"Score: {img_info['similarity_score']:.4f}")
                    self.results_grid.addWidget(score_label, (i // 3) + 1, i % 3)

def main():
    app = QApplication(sys.argv)
    window = ImageRetrievalGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()