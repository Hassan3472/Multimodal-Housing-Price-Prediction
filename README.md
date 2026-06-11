# Multimodal Housing Price Prediction

This project demonstrates a multimodal approach to predicting housing prices by combining tabular data features with visual features extracted from house images.

## Project Overview

The goal of this project is to build a machine learning model that leverages both structured (tabular) data and unstructured (image) data to enhance housing price prediction accuracy. We use a pre-trained Convolutional Neural Network (MobileNetV2) for image feature extraction and traditional machine learning models (Random Forest, XGBoost) for regression.

## Table of Contents

1.  [Setup and Data Loading](#setup-and-data-loading)
2.  [Data Loading & Validation](#data-loading--validation)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Data Preprocessing](#data-preprocessing)
5.  [CNN Feature Extraction](#cnn-feature-extraction)
6.  [Multimodal Feature Fusion](#multimodal-feature-fusion)
7.  [Model Training](#model-training)
8.  [Hyperparameter Tuning](#hyperparameter-tuning)
9.  [Evaluation](#evaluation)
10. [Best Model Selection](#best-model-selection)
11. [Export Model](#export-model)

## 1. Setup and Data Loading

This section covers importing necessary libraries and loading the datasets. The tabular data used is a modified version of the California Housing Prices dataset. For image data, dummy images are generated due to challenges in programmatically downloading a large, consistent public dataset.

## 2. Data Loading & Validation

-   **Tabular Data:** Loaded and inspected for structure, missing values, and data types. Missing values in `total_bedrooms` and `ocean_proximity` (categorical) are identified.
-   **Image Data:** 10 dummy `224x224x3` images are generated to simulate visual input, allowing the multimodal pipeline to proceed.

## 3. Exploratory Data Analysis (EDA)

Minimal EDA is performed on the tabular data, including:
-   Distribution of housing prices.
-   Relationships between `total_bedrooms`, `total_rooms` (as a proxy for size), and `median_house_value`.
-   A correlation heatmap revealing `median_income` as a strong predictor.

## 4. Data Preprocessing

-   **Tabular Data:** Split into training and testing sets. A `ColumnTransformer` handles numerical feature imputation (median) and scaling (`StandardScaler`), and categorical feature imputation (most frequent) and one-hot encoding (`OneHotEncoder`).
-   **Image Data:** Images are loaded into a `tf.data.Dataset`, resized to `224x224`, decoded, and normalized using `mobilenet_v2.preprocess_input`.

## 5. CNN Feature Extraction

A pre-trained `MobileNetV2` model (without its top classification layer) is used as an image feature extractor. Its base layers are frozen, and a `GlobalAveragePooling2D` layer is added to generate image embeddings. Features are extracted for the dummy images, resulting in `(10, 1280)` dimensional vectors.

## 6. Multimodal Feature Fusion

The extracted image features are concatenated with the preprocessed tabular features. For demonstration, the first 10 rows of processed tabular data are combined with the 10 image features, creating a fused multimodal feature set of shape `(10, 1293)` for both training and testing.

## 7. Model Training

Two regression models are trained on the fused multimodal data:
-   **Random Forest Regressor:** Initial performance was MAE: 138055.81, RMSE: 157256.39, R²: -0.17.
-   **XGBoost Regressor:** Initial performance was MAE: 108238.78, RMSE: 120911.61, R²: 0.31.

## 8. Hyperparameter Tuning

`GridSearchCV` was used to tune the `RandomForestRegressor` with a small parameter grid (`n_estimators`: [50, 100], `max_depth`: [10, 20], `min_samples_split`: [2, 5]).
-   Best Parameters: `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 50}`
-   Tuned RF Performance: MAE: 139297.73, RMSE: 158002.87, R²: -0.18.

## 9. Evaluation

Model performance is evaluated through:
-   **Actual vs. Predicted Scatter Plots:** Visual comparison of model predictions against true values for each model.
-   **Model Performance Comparison Table:** A summary table displaying MAE, RMSE, and R² scores for all models (Original RF, XGBoost, Tuned RF).

## 10. Best Model Selection

The best performing model is automatically identified based on the lowest RMSE.
-   **Selected Model:** `XGBoostRegressor`
-   **Performance:** MAE: 108238.78, RMSE: 120911.61, R² Score: 0.31

## 11. Export Model

The best model (`XGBoostRegressor`) and the tabular data preprocessor (`ColumnTransformer`) are saved using `joblib`. A demonstration is provided to load these saved components and make predictions on new, simulated data, showcasing the end-to-end inference process.

```python
# Example of loading and predicting
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessor
loaded_model = joblib.load('best_multimodal_model.joblib')
loaded_preprocessor = joblib.load('tabular_preprocessor.joblib')

# Simulate new unseen data (replace with actual new data in a real scenario)
# Example: one row of tabular data similar to the original dataset
example_tabular_data = pd.DataFrame([[-118.38, 34.17, 33.0, 1588.0, 454.0, 739.0, 392.0, 2.8208, '<1H OCEAN']],
                                     columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

# Preprocess tabular data
processed_tabular = loaded_preprocessor.transform(example_tabular_data)

# Generate dummy image features (in a real scenario, use your image_feature_extractor)
dummy_image_features = np.random.rand(1, 1280) # 1 sample, 1280 features

# Combine features
combined_input = np.concatenate((processed_tabular, dummy_image_features), axis=1)

# Make prediction
prediction = loaded_model.predict(combined_input)
print(f"Predicted House Value: ${prediction[0]:.2f}")
```

## How to Run the Notebook

1.  Open the `.ipynb` file in Google Colab.
2.  Run all cells sequentially. The notebook is designed to execute end-to-end.
3.  Ensure you have an active internet connection for downloading pre-trained model weights (MobileNetV2).

**Note on Image Data:** Due to the use of dummy images for demonstration, the model's performance metrics are illustrative rather than indicative of real-world accuracy with actual house image data.
