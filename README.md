# EigenBirkin

A comprehensive image classification project implementing both classical machine learning and deep learning approaches for image recognition and analysis.

## Overview

EigenBirkin is a machine learning project that explores image classification using multiple approaches:
- **PCA-based Classification**: Eigenfaces/eigenvalues approach using Principal Component Analysis
- **K-Nearest Neighbors (KNN)**: Classical ML classifier with sensitivity analysis
- **Convolutional Neural Networks (CNN)**: Deep learning approach for comparison

The project demonstrates the effectiveness of dimensionality reduction techniques combined with traditional ML algorithms, while also benchmarking against modern deep learning methods.

## Features

- **Eigenface Generation**: Creates principal components from image datasets for efficient representation
- **KNN Classification**: Implements k-nearest neighbors with configurable parameters
- **Parameter Sensitivity Analysis**: Explores how PCA components and KNN parameters affect performance
- **CNN Implementation**: Neural network baseline for comparison
- **Data Processing Pipeline**: Automated data loading, preprocessing, and augmentation
- **Performance Testing**: Comprehensive accuracy and performance evaluation

## Project Structure

```
EigenBirkin/
├── eigenbag_generator.py       # Generates eigenfaces/principal components
├── knn_classifier.py           # KNN classification implementation
├── pca_knn_sensitivity.py      # Tests parameter sensitivity
├── cnn_classification_nn.ipynb # CNN implementation and comparison
├── create_test_dataset.py      # Generates/prepares test datasets
├── data_mixer.py               # Data preprocessing and augmentation
├── run_pipeline.py             # Main pipeline orchestrator
├── test_knn_accuracy.py        # Accuracy testing and evaluation
├── Data/                       # Input datasets
├── outputs/                    # Generated results and models
└── layers/                     # Saved model layers/components
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- Scikit-learn
- PyTorch or TensorFlow (for CNN)
- Pandas
- Matplotlib (for visualization)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/thechakmv/EigenBirkin.git
cd EigenBirkin
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy scikit-learn torch pandas matplotlib
```

## Usage

### Running the Full Pipeline

Execute the complete classification pipeline:
```bash
python run_pipeline.py
```

This will:
1. Load and preprocess your dataset
2. Generate eigenfaces using PCA
3. Train KNN classifiers
4. Run CNN for comparison
5. Output results and metrics

### Individual Components

#### Generate Eigenfaces
```bash
python eigenbag_generator.py
```

#### Train KNN Classifier
```bash
python knn_classifier.py
```

#### Analyze Parameter Sensitivity
```bash
python pca_knn_sensitivity.py
```

#### Test Accuracy
```bash
python test_knn_accuracy.py
```

#### Create Custom Test Dataset
```bash
python create_test_dataset.py
```

#### Preprocess Data
```bash
python data_mixer.py
```

## How It Works

### Eigenface Approach
1. **Dimensionality Reduction**: PCA is applied to flatten image datasets, extracting the most significant features (eigenfaces)
2. **Efficient Representation**: Images are projected onto these principal components, reducing dimensionality while preserving important information
3. **KNN Classification**: The reduced-dimensional representations are used with k-nearest neighbors for classification

### CNN Baseline
Convolutional neural networks are trained for direct comparison to demonstrate the effectiveness of the eigenface approach.

## Results

The project includes comprehensive evaluation metrics:
- Classification accuracy on test datasets
- Parameter sensitivity analysis (number of PCA components vs. performance)
- K-value sensitivity for KNN
- Comparison between classical ML (PCA+KNN) and deep learning (CNN) approaches

Results are saved in the `outputs/` directory.

## Key Parameters

- **n_components**: Number of principal components to retain (affects accuracy and speed)
- **k**: Number of neighbors in KNN classifier
- **train_test_split**: Ratio for training/testing data division
- **random_state**: Seed for reproducibility

## Contributing

Contributions are welcome! Please feel free to:
- Submit issues for bugs or suggestions
- Fork and create pull requests with improvements
- Propose new classification approaches
- Add support for additional datasets

## License

This project is open source and available under the MIT License.

## Author

Created by [Aaron Chakma](https://github.com/thechakmv)

## Contact

For questions or collaboration opportunities, please reach out via GitHub issues or email.

---

**Note**: The project name "EigenBirkin" is a creative reference combining "Eigen" (eigenvalues/eigenvectors from PCA) with a fashion reference, reflecting the intersection of mathematical principles and creative application in machine learning.
