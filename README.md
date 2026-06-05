# Brain Tumor Localization & Classification Pipeline (BraTS-PEDs)

A complete computer vision and machine learning pipeline for brain tumor processing, feature extraction, traditional classification, and deep learning-based localization (YOLOv8) using the **BraTS-PEDs (Pediatric Brain Tumor) dataset**.

This repository contains the full source code and rendered outputs for the project.

---

## Repository Files

* **[CV_Eproject_Fixed_v2_final (1).ipynb](file:///C:/Users/lizaa/Downloads/CV_Eproject_Fixed_v2_final%20%281%29.ipynb)**: The primary, cleaned Jupyter Notebook containing the executable code cells (outputs cleared to keep file size under 1MB).
* **[CV_Eproject_Fixed_v2_final.html](file:///C:/Users/lizaa/Downloads/CV_Eproject_Fixed_v2_final.html)**: The fully rendered HTML backup of the notebook containing all visualizations, logs, training curves, and output tables.

---

## Pipeline Architecture

### 1. Preprocessing & Quality Control (E-Assignment 1)
* **Unsupervised Slice Selection**: Automatically identifies the 2D axial slice with the highest intensity variance of brain tissue. Implements a self-calibrating intensity threshold and limits the search range (15%–80% of axial volume) to prevent selecting empty neck or top-scalp slices.
* **Radiometric Resolution (Bit-depth Check)**: Detects whether input modalities (`t1n`, `t1c`, `t2w`, `t2f`) are 8-bit, 16-bit, or 32-bit to prevent false contouring during processing.
* **Filtering & Denoising**: Applies and compares Gaussian, Mean, and Median filters, quantifying restoration quality using **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index).
* **Anti-Aliasing Downsampling**: Downsamples slices from $240 \times 240$ to $120 \times 120$ utilizing a Gaussian pre-filter to prevent aliasing/checkerboard artifacts.

### 2. Segmentation & Shape Representation (E-Assignment 2)
* **Edge Detection**: Applies Canny and Sobel edge detectors on the tumor region of interest.
* **Morphological Cleaning**: Bridges boundary gaps and cleans noise by comparing multiple morphological operations. **Binary Morphological Closing** is selected as the Master Mask for downstream tasks.
* **Quality Control Gate**: Implements a strict QC pipeline. Slices with tumor areas $< 50$ pixels on the $120 \times 120$ grid, failed contour extractions, or failed convex hulls are automatically logged and dropped from subsequent steps.
* **Boundary Descriptors**: Traces tumor boundaries to extract 8-directional **Absolute Chain Codes**, computes the **First Difference** (rotation invariance), and normalizes to a **Shape Number** (starting-point invariance).
* **Computational Geometry**: Calculates the **Convex Hull** of the tumor boundary to measure the minimal convex bounding polygon, tracking the hull area and perimeter.

### 3. Feature Engineering & Traditional ML (E-Assignment 3)
* **Gray-Level Co-occurrence Matrix (GLCM)**: Quantizes tumor intensities to 16 gray levels and calculates GLCM at $0^\circ$ orientation and a distance of 1 pixel.
* **Feature Vector Compilation**: Computes a comprehensive feature vector for each active slice, saving the metrics (`Area`, `Perimeter`, `Circularity`, `Centroid`, `GLCM Energy`, `GLCM Contrast`, and `GLCM Entropy`) to `BraTS_Complete_Feature_Vector.csv`.
* **Unsupervised Labeling**: Uses **K-Means Clustering** (k=2) to automatically label tumors. Slices with high entropy and low circularity are labeled **Malignant** (irregular/chaotic), while slices with low entropy and high circularity are labeled **Benign** (smooth/uniform).
* **Traditional Classifier**: Trains a **K-Nearest Neighbors (KNN)** classifier (k=3) on the scaled feature vectors using **5-Fold Stratified Cross-Validation** to predict malignancy.

### 4. Deep Learning Localization (Final Project)
* **YOLOv8 Dataset Preparation**: Automatically converts morphological masks to normalized YOLO bounding box formats (`class x_center y_center width height`).
* **Deep Learning Pipeline**: Trains a **YOLOv8s** object detection model on CPU using the $120 \times 120$ downsampled, anti-aliased slices directly (without specular mitigation) for 50 epochs.
* **Evaluation**: Generates loss curves, precision-recall metrics, validation bounding box overlays, and a normalized confusion matrix. Includes analysis explaining single-class detection metrics (e.g., why True Negatives are blank and background false positives scale).

---

## Requirements & Installation

To run the notebook locally, ensure you have the following packages installed:

```bash
pip install numpy pandas matplotlib scipy scikit-image scikit-learn nibabel opencv-python ultralytics
```

---

##  Performance Summary
A total summary of all dropped slices and model metrics (such as the 5-Fold KNN cross-validation scores and YOLOv8 training loss curves) are embedded and fully readable in the accompanying [CV_Eproject_Fixed_v2_final.html](file:///C:/Users/lizaa/Downloads/CV_Eproject_Fixed_v2_final.html) report.
