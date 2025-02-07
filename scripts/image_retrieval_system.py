import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QLineEdit, QFrame)
from PyQt5.QtCore import Qt


class ImageRetrievalGUI(QMainWindow):
    def __init__(self, W, combined_df):
        super().__init__()
        self.W = W
        self.combined_df = combined_df
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Retrieval')
        self.setGeometry(400, 200, 400, 150)
        self.setStyleSheet('background-color: black;')  # color

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(40, 20, 40, 20)

        # create a container for centered content
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # input section with styling
        self.input_label = QLabel('Enter image ID or index:')
        self.input_label.setAlignment(Qt.AlignCenter)  
        self.input_label.setStyleSheet('font-size: 14px; margin-bottom: 5px; color: white;')
        
        self.input_field = QLineEdit()
        self.input_field.setFixedHeight(30)
        self.input_field.setStyleSheet('padding: 5px; border: 1px solid #ccc; border-radius: 4px; color: white;')
        
        self.search_button = QPushButton('Search')
        self.search_button.setFixedHeight(35)
        self.search_button.setStyleSheet('''
            QPushButton {
                background-color: white;
                color: black;
                border: none;
                border-radius: 4px;
                padding: 5px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: grey;
            }
        ''')
        self.search_button.clicked.connect(self.search_images)

        # add widgets to container
        container_layout.addWidget(self.input_label)
        container_layout.addWidget(self.input_field)
        container_layout.addWidget(self.search_button)
        container_layout.setSpacing(10)

        # add container to main layout
        layout.addWidget(container)

    def search_images(self):
        query = self.input_field.text()
        try:
            query = int(query)
        except ValueError:
            query = query.strip()
        
        query_image(self.W, self.combined_df, query, top_k=5, image_folder='data/images')


def load_saved_model(hypergraph_file='hypergraph_data.npz', metadata_file='combined_data.csv'):
    if not os.path.exists(hypergraph_file):
        print(f"Saved hypergraph model '{hypergraph_file}' not found. Please run the model-building script first.")
        exit(1)
    data = np.load(hypergraph_file)
    W = data['W']
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file '{metadata_file}' not found. Please ensure you have saved the combined metadata.")
        exit(1)
    combined_df = pd.read_csv(metadata_file)
    
    return W, combined_df

def query_image(W, combined_df, query, top_k=5, image_folder='images'):
    # Determine query index: if query is int, use it directly; otherwise, search for the image_id.
    if isinstance(query, int):
        query_index = query
    else:
        matches = combined_df[combined_df['image_id'] == query].index
        if len(matches) == 0:
            print(f"No image with id '{query}' found.")
            return
        query_index = matches[0]
    
    # Retrieve top_k similar images using the saved affinity matrix W.
    # Exclude the query image itself.
    sorted_indices = np.argsort(W[query_index])[-top_k*2:][::-1]
    top_indices = [idx for idx in sorted_indices if idx != query_index][:top_k]
    
    query_meta = combined_df.iloc[query_index]
    print(f"Top {top_k} similar images for image {query_meta['image_id']}:")
    for rank, idx in enumerate(top_indices):
        score = W[query_index, idx]
        meta = combined_df.iloc[idx]
        print(f"  {rank + 1}. Image ID: {meta['image_id']}, "
              f"Class ID: {meta['class_id']}, "
              f"Species: {meta['species']}, "
              f"Breed ID: {meta['breed_id']}, "
              f"Score: {score:.4f}")
    
    precision = evaluate_precision(W, combined_df, query_index, k=top_k)
    print(f"\nPrecision@{top_k}: {precision:.4f}")

    # Plot the query image and its top similar images.
    # We assume each image is named as image_id + ".jpg" in the specified image_folder.
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['figure.edgecolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['lines.color'] = 'white'
    num_plots = top_k + 1
    fig, axes = plt.subplots(2, num_plots, figsize=(4 * num_plots, 8))
    fig.patch.set_facecolor('black')

    # Query image
    query_meta = combined_df.iloc[query_index]
    query_image_path = os.path.join(image_folder, f"{query_meta['image_id']}.jpg")
    try:
        query_img = Image.open(query_image_path)
        axes[0,0].imshow(query_img)
    except Exception as e:
        print(f"Could not load query image: {e}")
    axes[0,0].axis('off')
    axes[0,0].set_title("Query", color='white')
    axes[0,0].set_facecolor('black')

    # Query metadata
    meta_text = f"Image ID: {query_meta['image_id']}\nClass ID: {query_meta['class_id']}\n" \
                f"Species: {query_meta['species']}\nBreed ID: {query_meta['breed_id']}"
    axes[1,0].text(0.5, 0.5, meta_text, ha='center', va='center', color='white')
    axes[1,0].axis('off')
    axes[1,0].set_facecolor('black')

    # Similar images
    for i, idx in enumerate(top_indices, start=1):
        meta = combined_df.iloc[idx]
        score = W[query_index, idx]
        
        # Image
        image_path = os.path.join(image_folder, f"{meta['image_id']}.jpg")
        try:
            img = Image.open(image_path)
            axes[0,i].imshow(img)
        except Exception as e:
            print(f"Could not load image: {e}")
        axes[0,i].axis('off')
        axes[0,i].set_title(f"Rank {i}\nScore: {score:.4f}", color='white')
        axes[0,i].set_facecolor('black')
        
        # Metadata
        meta_text = f"Image ID: {meta['image_id']}\nClass ID: {meta['class_id']}\n" \
                   f"Species: {meta['species']}\nBreed ID: {meta['breed_id']}"
        axes[1,i].text(0.5, 0.5, meta_text, ha='center', va='center', color='white')
        axes[1,i].axis('off')
        axes[1,i].set_facecolor('black')

    plt.suptitle(f"Precision@{top_k}: {precision:.4f}", y=0.98, color='white')
    plt.tight_layout()
    plt.show()


def evaluate_precision(W, combined_df, query_index, k=5):
    """
    Evaluate precision@k for retrieval results
    """
    # Get ground truth: images of same breed/class as query
    query_meta = combined_df.iloc[query_index]
    ground_truth = combined_df[
        (combined_df['breed_id'] == query_meta['breed_id']) & 
        (combined_df.index != query_index)
    ].index.tolist()
    
    # Get top k retrieved images
    sorted_indices = np.argsort(W[query_index])[-k*2:][::-1]
    retrieved = [idx for idx in sorted_indices if idx != query_index][:k]
    
    # Calculate precision
    relevant = sum(1 for idx in retrieved if idx in ground_truth)
    precision = relevant / k
    
    return precision


if __name__ == '__main__':
    # Load saved hypergraph model and metadata.
    # W, combined_df = load_saved_model()

    # # Ask user for a query: can be an image id or an index.
    # user_input = input("Enter image id or index to query: ")
    # try:
    #     query = int(user_input)
    # except ValueError:
    #     query = user_input.strip()
    
    # query_image(W, combined_df, query, top_k=5, image_folder='data/images')

    app = QApplication(sys.argv)
    W, combined_df = load_saved_model()
    window = ImageRetrievalGUI(W, combined_df)
    window.show()
    sys.exit(app.exec_())