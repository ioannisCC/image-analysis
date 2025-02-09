# Image Retrieval System Using Hypergraph-Based Manifold Ranking

This project implements an image retrieval system using hypergraph-based manifold ranking, as described in the paper *"Multimedia Retrieval through Unsupervised Hypergraph-based Manifold Ranking"* (IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 28, NO. 12, DECEMBER 2019). The system retrieves similar images based on content using a combination of feature extraction, hypergraph construction, and ranking algorithms.

![image](https://github.com/user-attachments/assets/41ff6de8-3d74-46b4-9595-ef1fd3ff9665)


---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Methodology](#methodology)
4. [Evaluation Results](#evaluation-results)
5. [Installation and Setup](#installation-and-setup)

---

## Project Overview

The goal of this project is to develop a content-based image retrieval system using hypergraph-based manifold ranking. The system extracts features from images using a pre-trained ResNet50 model, constructs a hypergraph to model relationships between images, and ranks images based on their similarity to a query image. Details on the project can be found in the documentation folder.

Key steps in the algorithm include:
1. **Feature Extraction**: Using ResNet50 to extract high-level features from images.
2. **Distance and Similarity Calculation**: Computing cosine distances and converting them to similarity scores.
3. **Rank Normalization**: Normalizing ranks based on reciprocal rank positions.
4. **Hypergraph Construction**: Building a hypergraph to model relationships between images.
5. **Hypergraph-Based Similarity**: Computing similarity matrices using hypergraph structures.
6. **Iterative Ranking**: Refining rankings iteratively (optional).
7. **Evaluation**: Measuring retrieval accuracy using precision, MAP, and NDCG.

---

## **Project Structure**

                                                ![image](https://github.com/user-attachments/assets/05c56dad-2b7f-428a-bf37-2f4c3a27d84e)

---

## **Methodology**
1. **Feature Extraction:**  
   - Using **ResNet50** to extract deep feature representations from each image.
   - Feature vectors capture meaningful visual information for similarity comparison.

2. **Similarity Calculation:**  
   - Compute pairwise **cosine similarity** between images to build an initial similarity matrix.

3. **Ranking and Refinement:**  
   - Use a **hypergraph-based ranking approach** to capture local and global relationships between images.
   - Optionally, apply **iterative refinement** to update rankings using feedback.

4. **Evaluation:**  
   - Evaluate the effectiveness using **Precision@k**, **Mean Average Precision (MAP)**, and **NDCG** metrics.

---

## **Evaluation Results**
| Metric        | Top-5   |
|---------------|---------|
| Precision     | 0.6929  |
| MAP           | 0.7274  |
| NDCG          | 0.7383  |

---

## **Installation and Setup**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ioannisCC/image-analysis.git
   cd image-analysis

2. **Create a virtual environment and install dependencies:**
  *For Linux/MacOS:*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
  *For Windows:*
    ```bash
  python -m venv venv
  venv\Scripts\activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

4. **Prepare the Dataset:**
  - Download the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
  - Extract the dataset and place the images and annotations folders inside the data directory of the project (like in the image above).

5. **Extract Features:**
   **Run the feature extraction script to extract features from the images using the ResNet50 model:**
   ```bash
   python scripts/final_working_scripts/feature_extraction.py
   
  This will generate a file called features.npy and another one all_features.pkl in the artifacts folder.
  
6. **Run the algorithm:**
   **Run the manifold ranking script to build the hypergraph and compute similarity matrices:**
   ```bash
   python scripts/final_working_scripts/manifold_ranking.py
   
**This will generate the following files in the artifacts folder:**

  - hypergraph_data.npz: Contains hypergraph data (H, Sh, Sv, W).

  - combined_data.csv: Contains metadata and feature vectors for all images.

7. **Run the Image Retrieval System in GUI**
   **To retrieve similar images for a query image, run the following script:**
   ```bash
   python scripts/final_working_scripts/image_retrieval_system.py

The GUI allows you to:
  
  * Enter an image ID or index.
  
  * View the top-k similar images.
  
  * Display metadata and similarity scores.

## **Notes**

- Ensure you have sufficient computational resources (e.g., GPU) for faster processing, especially during feature extraction.

- Modify the scripts/final_working_scripts/manifold_ranking.py script to adjust hyperparameters (e.g., k, L) for better results.

- The artifacts folder stores precomputed data to avoid redundant computations. Delete these files if you want to recompute them.
