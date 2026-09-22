# Real Estate Rent Price Prediction: ML and Deep Learning Pipeline

An end-to-end Machine Learning and Deep Learning pipeline that collects real-world rental listings via web scraping, performs exploratory data analysis and feature engineering, and predicts rental prices using Scikit-Learn models and TensorFlow/Keras.

---

## Project Overview

The objective of this project is to automate the extraction of property listings and benchmark classical regression algorithms against deep learning architectures for rental price forecasting.

* **Data Extraction:** Scraped active listings using requests and BeautifulSoup.
* **Preprocessing and Feature Engineering:** Extracted numerical features (m2, room configurations), handled outliers, applied One-Hot Encoding for districts, and scaled features using StandardScaler.
* **Model Benchmarking:** Trained Linear Regression, Random Forest Regressor, and a Multi-Layer Perceptron (MLP) built with TensorFlow/Keras.

---

## Tech Stack and Libraries

* **Language:** Python 3.x
* **Web Scraping:** requests, beautifulsoup4
* **Data Manipulation and Analysis:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning:** scikit-learn
* **Deep Learning:** tensorflow, keras

---

## Model Architectures and Pipeline

1. **Feature Engineering:**
   * Parsed room syntax (e.g., 3+1 -> 4 total rooms).
   * Encoded categorical location attributes (ilce) using dummy variables.
   * Standardized input features with StandardScaler to stabilize gradient descent.

2. **Keras Neural Network Architecture:**
   * **Input Layer:** Scaled input features.
   * **Hidden Layers:**
     * Dense(128, ReLU) + Dropout(0.2)
     * Dense(64, ReLU) + Dropout(0.2)
     * Dense(32, ReLU)
   * **Output Layer:** Dense(1, Linear) for continuous price estimation.
   * **Optimization:** Adam optimizer with mean_squared_error loss and early stopping callbacks.

---

## Experimental Results

| Model | MAE (TL) | R2 Score |
| :--- | :---: | :---: |
| **Linear Regression** | — | **0.9755** |
| **Random Forest Regressor** | **1,953 TL** | **0.9920** |
| **Keras MLP (Deep Learning)** | **4,620 TL** | **0.9792** |

### Key Findings and Model Selection
* All models achieved strong predictive performance (R2 > 97%), verifying the quality of the feature representation.
* **Random Forest Regressor** delivered the best overall performance with the lowest Mean Absolute Error (1,953 TL) and highest variance explanation (R2 = 0.9920).
* While the **Keras MLP** model generalized well (R2 = 0.9792), ensemble decision tree algorithms naturally excel on structured tabular data of this scale without requiring massive training datasets. Therefore, **Random Forest** is selected as the primary production model.

---

## Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/rent-price-prediction.git](https://github.com/your-username/rent-price-prediction.git)
cd rent-price-prediction
