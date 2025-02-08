# Content-Based Image Retrieval with Hypergraph Ranking

This project implements a **content-based image retrieval system** using **hypergraph-based ranking and similarity refinement techniques**. The system retrieves and ranks visually similar images based on deep visual features extracted from the dataset using a **pre-trained ResNet50 model**.

---

## **Project Overview**
- **Objective:** To retrieve and rank similar images from a dataset using **visual content** instead of metadata.
- **Techniques Used:**  
  - Feature extraction using **ResNet50**  
  - **Cosine similarity** for initial ranking  
  - **Hypergraph-based ranking refinement**  
  - **Iterative ranking updates** (optional, experimental)  
  - Evaluation using metrics: **Precision@k**, **MAP**, and **NDCG**

---

## 📁 **Project Structure**

![image](https://github.com/user-attachments/assets/05c56dad-2b7f-428a-bf37-2f4c3a27d84e)

📦 assignment/
├── 📂 artifacts/
│   ├── 📄 all_features.pkl          # Extracted features with metadata
│   ├── 📄 combined_data.csv         # Combined metadata for retrieval
│   ├── 📄 features.npy              # ResNet50 feature vectors
│   └── 📄 hypergraph_data.npz       # Hypergraph model data
├── 📂 data/
│   ├── 📂 annotations/              # Dataset annotations
│   └── 📂 images/                   # Pet dataset images
├── 📂 documentation/                # Project documentation
├── 📂 scripts/
│   ├── 📂 drafts_&_tests/          # Development scripts
│   └── 📂 final_working_scripts/
│       ├── 📄 feature_extraction.py     # ResNet50 feature extraction
│       ├── 📄 image_retrieval_system.py # Main retrieval system
│       └── 📄 manifold_ranking.py       # Hypergraph ranking implementation
└── 📄 .gitignore

---

## 🔍 **Methodology**
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

## 📊 **Evaluation Results**
| Metric        | Top-5   |
|---------------|---------|
| Precision     | 0.6929  |
| MAP           | 0.7274  |
| NDCG          | 0.7383  |

---

## 🔧 **Installation and Setup**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ioannisCC/image-analysis.git
   cd image-analysis

2. 	Create a virtual environment and install dependencies:


