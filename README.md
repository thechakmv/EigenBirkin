# EigenBirkin: Birkin Bag Classification with PCA and KNN

A classical machine learning approach to binary image classification: distinguishing Hermès Birkin bags from other handbags using Principal Component Analysis (PCA) for dimensionality reduction and k-nearest neighbors (KNN) for classification.

## Overview

The Hermès Birkin is one of the world's most sought-after luxury handbags, with prices ranging from $7,000 to over $300,000. This project builds an **interpretable, lightweight baseline** for Birkin classification that emphasizes understanding over black-box deep learning.

**Key Results:**
- **95.59% test accuracy** on 337-image dataset (207 Birkin, 130 non-Birkin)
- Optimal configuration: 12 PCA components, k=3 KNN
- Focus on explainability: eigenbag visualizations, image reconstruction, sensitivity analysis

## Method

### 1. Data Preprocessing
- Convert all images to grayscale
- Resize to fixed 64×64 resolution (4,096-dimensional vectors)
- Encode labels: Birkin=1, non-Birkin=0

### 2. Eigenbag Feature Extraction (PCA)
PCA learns a low-dimensional orthonormal basis from the training set, capturing the dominant variance in Birkin vs. non-Birkin images:

```
z = U_k^T (x - μ)
```

Where:
- `x` = vectorized image
- `μ` = training set mean
- `U_k` = top k principal components
- `z` = low-dimensional projection

**Default configuration:** k=12 to k=48 principal components (sensitivity analyzed)

The learned "eigenbirkins" reveal:
- Silhouette and handle shape
- Bag size and pose variation
- Broad color/appearance patterns

See Figure 1 in the project report for visualizations of the mean Birkin image and top principal components.

### 3. Classification with Distance-Weighted KNN
After projection to the low-dimensional eigenbag space, classification uses k-nearest neighbors with distance weighting:
- For each test sample, find its k nearest training neighbors
- Predict the dominant class, weighting closer neighbors more heavily
- **Advantages:** Intuitive, parameter-efficient, no additional learned layers

**Evaluated k ∈ {1, 3, 5}** — small neighborhoods performed best.

### 4. Qualitative Reconstruction
Verify PCA captures meaningful structure by reconstructing images from their low-dimensional coefficients:

```
x̂ = U_k z + μ
```

Reconstructed images preserve main object shape and intensity while smoothing fine-grained details, confirming effective low-rank approximation.

## Dataset

**Composition:**
- **207 Birkin images** (positive class)
- **130 non-Birkin handbag images** (negative class)
- **Total: 337 images**

The updated dataset is more balanced and realistic than earlier iterations, making evaluation more informative.

## Results

### Test Performance (Optimal Configuration)
| Method | PCA Components | k | Accuracy | TN | FP | FN | TP |
|--------|---|---|----------|----|----|----|----|
| PCA + KNN | 12 | 3 | **95.59%** | 23 | 3 | 0 | 42 |

### Sensitivity Analysis
Configuration sweep over PCA dimension and KNN neighborhood size:

| PCA Comps | k | Accuracy | Details |
|-----------|---|----------|---------|
| 12 | 1 | 95.59% | Optimal (also k=3 with 12 comps) |
| 12 | 3 | 95.59% | ✓ Best overall |
| 12 | 5 | 91.18% | TN=21, FP=3, FN=2, TP=39 |
| 24 | 1 | 94.12% | TN=23, FP=2, FN=1, TP=41 |
| 24 | 3 | 91.18% | TN=21, FP=3, FN=2, TP=39 |
| 24 | 5 | 88.24% | TN=19, FP=5, FN=3, TP=38 |
| 48 | 1 | 86.76% | Over-fit on high-dimensional basis |
| 48 | 3 | 79.41% | High variance, less stable |
| 48 | 5 | 76.47% | Further degradation |

**Key insight:** Smaller PCA basis (12 components) generalizes better than larger bases, suggesting that a compact representation is more robust than preserving every direction of variation.

## Related Work

This project integrates three classical directions in computer vision and pattern recognition:

1. **Eigenfaces (Turk & Pentland):** PCA-based face recognition; we apply the same "eigenbags" concept to handbag images
2. **Non-parametric KNN (Cover & Hart):** Distance-weighted nearest neighbors for pattern recognition
3. **Convolutional Neural Networks (LeNet, AlexNet):** Modern hierarchical feature learning (explored in optional CNN extension notebook)

The project deliberately uses classical, interpretable methods over deep learning because they are easier to understand, reproduce, and debug on moderate-sized datasets.

## Technical Stack

- **Python 3**
- **NumPy** – matrix operations, image vectorization
- **Scikit-learn** – PCA (randomized SVD), KNN classifier
- **Matplotlib** – visualization (mean images, eigenbags, reconstructions)
- **Pillow** – image I/O and preprocessing

## Project Structure

```
EigenBirkin/
├── data/
│   ├── Birkin/           # Positive examples
│   ├── birkins/          # Additional positive examples
│   └── other/            # Negative examples (non-Birkin bags)
├── EigenBirkin.ipynb     # Main pipeline (PCA + KNN)
├── CNN_extension.ipynb   # Optional: neural network baseline
├── README.md
└── Final_Project_176.pdf # Full project report
```

## Quick Start

### 1. Prepare Dataset
Organize images into folders:
```
data/
├── Birkin/
├── birkins/
└── other/
```

### 2. Run the Pipeline
Open `EigenBirkin.ipynb` and execute cells sequentially:
- Load and preprocess images (grayscale, resize)
- Fit PCA on training split
- Train KNN classifier
- Evaluate on test set
- Visualize eigenbags and reconstructions

### 3. Sensitivity Analysis
Modify `n_components` and `k` parameters to explore the hyperparameter space shown in the results table.

## Key Findings

1. **12 PCA components is optimal** for this dataset, suggesting a compact representation generalizes better than preserving more dimensions
2. **Small k values (k=1, 3) outperform k=5**, indicating that local neighborhood structure is highly informative
3. **Zero false negatives** in the optimal configuration (42/42 Birkins correctly identified)
4. **Learned features are interpretable:** Eigenbags show recognizable bag features (handles, silhouettes, proportions)
5. **Classical methods are competitive** on small datasets; deep learning not necessary here

## Interpretability Advantages

Unlike black-box neural networks, this pipeline enables:
- **Visualization of learned features** (mean image, principal components, reconstructions)
- **Understanding of prediction basis** (which training neighbors contributed to a decision?)
- **Easy hyperparameter tuning** (manual sweep, no training epochs)
- **Reproducibility** (fully deterministic, no random initialization)

## Future Extensions

1. **CNN baseline:** `CNN_extension.ipynb` explores how deep learning performs on the same task
2. **Larger dataset:** Extend Birkin and non-Birkin classes with more images
3. **Color features:** Include color information instead of grayscale-only
4. **Transfer learning:** Fine-tune a pretrained ResNet or Vision Transformer
5. **Ensemble methods:** Combine PCA+KNN with other classifiers

## License

Academic project. See project report for attribution.

## Citation

```bibtex
@misc{eigenbirkin2025,
  title={EigenBirkin: Birkin Bag Classification with PCA},
  author={Anderson, Calvin and Chakma, Aaron},
  year={2025},
  note={ECE 176 Final Project, UC San Diego}
}
```

---

**Note:** This project demonstrates classical machine learning on a interpretable, moderate-sized dataset. For large-scale image classification, modern deep learning approaches (CNNs, Vision Transformers) typically outperform PCA+KNN. The value here is **understanding the fundamentals** and achieving strong results with transparent, debuggable methods.
