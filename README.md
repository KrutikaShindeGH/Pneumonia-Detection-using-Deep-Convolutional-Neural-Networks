# Pneumonia-Detection-using-Deep-Convolutional-Neural-Networks
A comprehensive deep learning project for detecting pneumonia from chest X-ray images using a custom-built Deep Convolutional Neural Network (DCNN).

## 📋 Overview

This project implements an end-to-end medical image classification system that analyzes chest X-rays to detect pneumonia. The model achieves high accuracy in distinguishing between normal and pneumonia cases, providing risk scores and confidence levels for each prediction.

## 🎯 Key Features

- **Deep CNN Architecture**: 4-layer convolutional neural network with batch normalization and dropout
- **Data Augmentation**: Rotation, shifting, zooming, and flipping to improve model generalization
- **Risk Assessment**: Probability-based risk scoring (Low/Medium/High risk categories)
- **Comprehensive Visualization**: Training history, confusion matrices, ROC curves, and patient-level predictions
- **Detailed Reporting**: CSV exports with patient-level predictions and model performance metrics
- **Model Persistence**: Saved trained models for deployment and future use

## 🏗️ Model Architecture

```
- Conv Block 1: 2x Conv2D(32) + BatchNorm + MaxPool + Dropout
- Conv Block 2: 2x Conv2D(64) + BatchNorm + MaxPool + Dropout
- Conv Block 3: 2x Conv2D(128) + BatchNorm + MaxPool + Dropout
- Conv Block 4: 2x Conv2D(256) + BatchNorm + MaxPool + Dropout
- Dense Layer: 512 units + BatchNorm + Dropout
- Dense Layer: 256 units + BatchNorm + Dropout
- Output Layer: 1 unit (Sigmoid activation)
```

**Total Parameters**: ~6.5 million trainable parameters

## 📊 Dataset

**Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Dataset Statistics
- **Training Set**: 5,216 images
- **Validation Set**: 16 images
- **Test Set**: 624 images
- **Classes**: Normal (0), Pneumonia (1)

## 🚀 Getting Started

### Prerequisites

```python
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV (cv2)
```

### Installation

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
```

### Dataset Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract the `chest_xray.zip` file
3. Update the directory paths in the script:

```python
TRAIN_DIR = '/path/to/chest_xray/train'
VAL_DIR = '/path/to/chest_xray/val'
TEST_DIR = '/path/to/chest_xray/test'
```

### Running the Code

```bash
python csest_x_ray.py
```

Or run in Google Colab:
1. Upload the notebook to Colab
2. Mount Google Drive
3. Update the dataset paths
4. Run all cells

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224x224 pixels |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Binary Crossentropy |
| Early Stopping Patience | 10 epochs |
| Learning Rate Reduction | Factor 0.5, Patience 5 |

## 🎨 Visualizations Generated

The script automatically generates the following visualizations:

1. **sample_xrays.png** - Sample normal vs pneumonia X-rays
2. **grayscale_processing_visualization.png** - Image preprocessing steps
3. **vectorization_visualization.png** - Image-to-array conversion
4. **batch_processed_images.png** - Batch of processed images
5. **statistical_analysis.png** - Dataset statistical analysis
6. **model_architecture.png** - CNN architecture diagram
7. **training_history.png** - Loss, accuracy, AUC, precision/recall curves
8. **confusion_matrix.png** - Model performance confusion matrix
9. **roc_curve.png** - ROC curve with AUC score
10. **risk_score_distribution.png** - Risk score distributions
11. **risk_score_categories.png** - Risk category breakdown
12. **sample_predictions.png** - Sample patient predictions

## 📁 Output Files

### Patient Results Directory: `pneumonia_detection_patient_details/`

- **all_patient_results_[timestamp].csv** - Complete predictions for all test patients
- **model_summary_statistics_[timestamp].csv** - Overall model performance metrics
- **incorrect_predictions_[timestamp].csv** - Cases where model made errors
- **high_risk_patients_[timestamp].csv** - Patients classified as high-risk
- **false_negatives_CRITICAL_[timestamp].csv** - Missed pneumonia cases (critical for review)
- **training_history_[timestamp].csv** - Epoch-by-epoch training metrics

### Model Directory: `pneumonia_trained_models/`

- **pneumonia_detection_model_[timestamp].h5** - Trained model file
- **model_metadata_[timestamp].csv** - Model configuration and performance
- **best_pneumonia_model.h5** - Best model checkpoint during training

## 📊 Risk Score Interpretation

| Risk Score | Risk Level | Interpretation |
|------------|------------|----------------|
| 0% - 30% | Low Risk | Likely normal X-ray |
| 30% - 70% | Medium Risk | Uncertain - requires review |
| 70% - 100% | High Risk | Likely pneumonia detected |

## 🎯 Model Performance Metrics

The model tracks the following metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: Positive predictive value
- **Recall/Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## 🔍 Key Functions

### Data Processing
- `validate_directory()` - Validates dataset structure
- `load_and_preprocess_image()` - Loads and normalizes images
- `load_dataset()` - Loads complete dataset with labels

### Model
- `build_dcnn_model()` - Constructs the CNN architecture

### Risk Assessment
- `interpret_risk()` - Categorizes risk level from probability
- `get_confidence_level()` - Calculates prediction confidence

## 🚨 Critical Cases

The model specifically identifies and exports **False Negatives** (pneumonia cases classified as normal) to the file `false_negatives_CRITICAL_[timestamp].csv`. These cases require immediate medical review as they represent missed diagnoses.

## 📝 CSV Output Format

### Patient Results Columns
- Patient_ID, Index, True_Label, Predicted_Label
- Pneumonia_Probability, Normal_Probability
- Risk_Score_Percentage, Risk_Level
- Confidence_Level, Prediction_Status, Prediction_Type

## 🔄 Data Augmentation

Training images undergo the following augmentations:
- Rotation: ±15 degrees
- Width/Height Shift: ±10%
- Zoom: ±10%
- Horizontal Flip: Random
- Fill Mode: Nearest neighbor

