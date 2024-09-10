<p>
  # Parkinson's Disease Detection

This project is focused on detecting **Parkinson's disease** using machine learning algorithms. The dataset contains biomedical voice measurements from people with and without Parkinson's disease, and the machine learning model is trained to differentiate between healthy and affected individuals based on these features.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## About the Project

Parkinson's disease is a progressive nervous system disorder that affects movement. Early diagnosis is crucial for better treatment outcomes. This project utilizes machine learning to predict the presence of Parkinson’s disease based on voice data and other clinical measurements.

The goal of the project is to develop a model that can detect whether a person has Parkinson's disease by analyzing various vocal features extracted from recordings of their speech.

## Features

- **Machine Learning Model**: Uses classification algorithms to predict Parkinson's disease.
- **Feature Engineering**: Processes biomedical voice data to extract features.
- **Evaluation Metrics**: Calculates accuracy, precision, recall, and F1-score to measure model performance.
- **Cross-Validation**: Implements k-fold cross-validation to ensure the robustness of the model.

## Technologies Used

- **Python**: Main programming language used for model training and evaluation.
- **scikit-learn**: For machine learning algorithms such as SVM, Random Forest, etc.
- **pandas**: Data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.
- **Jupyter Notebook**: For writing and running the code interactively.

## Dataset

The dataset used for this project is the **Parkinson's Disease Data Set**, which contains the following features:

- **MDVP:Fo(Hz)** – Fundamental frequency of voice
- **MDVP:Fhi(Hz)** – Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)** – Minimum vocal fundamental frequency
- **Shimmer** – Variation in amplitude
- **Jitter** – Variation in frequency
- **NHR** – Noise-to-harmonics ratio
- **PPE** – Prosodic features
- **status** – The label (1 for Parkinson’s, 0 for healthy)

The dataset is available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons).

## Model Training

### 1. Data Preprocessing
- The dataset is loaded and preprocessed to handle missing values (if any) and scale features for better performance with machine learning algorithms.
- Feature selection is performed to identify the most important features contributing to the diagnosis.

### 2. Model Building
- A variety of machine learning models are built and tested, including:
  - **Support Vector Machines (SVM)**
  - **Random Forest Classifier**
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**

### 3. Model Evaluation
- The models are evaluated using cross-validation and the following metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- **Confusion Matrix** and **ROC curve** are also plotted for performance analysis.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- **Python 3.x**
- **pip** (Python package installer)
- **Jupyter Notebook** (for interactive development)


</p>
