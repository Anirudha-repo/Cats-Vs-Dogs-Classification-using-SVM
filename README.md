# Cats vs. Dogs Image Classification with SVM 🐾

## Project Overview

This repository contains a machine learning project focused on classifying images of cats and dogs. The goal is to build an accurate model that can distinguish between these two animals, leveraging traditional machine learning techniques for efficient and robust prediction.

##  Features

  * **Binary Image Classification:** Classifies input images into `Cat` or `Dog`.
  * **Support Vector Machine (SVM):** Utilizes a robust SVM classifier for accurate predictions.
  * **Dimensionality Reduction:** Employs Principal Component Analysis (PCA) to reduce feature complexity and improve performance.
  * **Standard Preprocessing:** Includes image resizing, flattening, and feature scaling for consistent data handling.
  * **API for Predictions:** (Optional) A Flask API endpoint is provided for real-time inference.

-----

##  Technologies Used

  * **Python:** Core programming language.
  * **scikit-learn:** Machine learning algorithms (SVM, PCA, MinMaxScaler).
  * **OpenCV (`cv2`):** For image processing operations.
  * **Pillow (`PIL`):** For image handling.
  * **NumPy:** For numerical computations.
  * **Matplotlib:** For data visualization.
  * **Flask:** (Optional) For creating the web API.

-----

##  Dataset

The project uses a dataset consisting of images of cats and dogs. Images are typically preprocessed (e.g., resized to 64x64 pixels) to ensure uniform input for the model.
You can download the complete dataset from Kaggle.

-----

##  Methodology

1.  **Image Loading & Preprocessing:** Images are loaded, resized to a consistent dimension, and then **flattened** into 1D arrays of pixel values.
2.  **Feature Scaling:** `MinMaxScaler` is applied to normalize pixel values, preventing features with larger ranges from dominating.
3.  **Dimensionality Reduction:** **Principal Component Analysis (PCA)** transforms the high-dimensional image data into a lower-dimensional space, capturing most of the variance while reducing computational load and noise.
4.  **Model Training:** A **Support Vector Machine (SVM)** classifier is trained on the processed, PCA-transformed data.
5.  **Evaluation:** The model's performance is assessed using standard classification metrics, including accuracy and the **Receiver Operating Characteristic (ROC) curve**.

-----

## ▶ How to Run Locally

Follow these steps to set up and run the project on your machine:

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2\. Set Up Virtual Environment (Recommended)

```bash
python -m venv env
# On Windows
.\env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

*Create a `requirements.txt` file in your project's root directory with the following contents:*

```
scikit-learn
opencv-python
Pillow
numpy
matplotlib
Flask # Only if you include app.py
```

### 4\. Prepare the Dataset

  * Download your Cats vs. Dogs dataset.
  * Place the image files in a structured way that your notebook can access (e.g., in a `dataset/` folder within your project). **Ensure the path to your dataset specified in your Jupyter notebook is correct.**

### 5\. Run the Jupyter Notebook

  * Start Jupyter Notebook from your project root:
    ```bash
    jupyter notebook
    ```
  * Open `Notebook (4).ipynb` (or your main notebook file).
  * Run all cells to execute the full pipeline: data loading, preprocessing, model training, and evaluation.

### 6\. (Optional) Run the Flask API for Predictions

If you plan to deploy your model via a web API:

  * Ensure your trained model components (`.pkl` files for SVC, MinMaxScaler, PCA) are saved in a directory like `deployed_model_cats_dogs/` within your project.
  * Run the Flask application from your terminal:
    ```bash
    python app.py
    ```
  * The API will typically be available at `http://127.0.0.1:5000`. You can test the `/predict` endpoint using tools like Postman by sending image data.

-----

