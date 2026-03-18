# 🏎️ Supervised Learning Project — Pre-Owned Cars

A comprehensive end-to-end Machine Learning pipeline analyzing the Indian pre-owned car market. This project implements advanced Data Preprocessing, Feature Scaling, Regression Modeling (Predicting Prices), and future Classification Modeling (Predicting Insurance Status) directly from raw CSV data.

---

## 📑 Table of Contents
1. [Project Objectives](#-project-objectives)
2. [Dataset Information](#-dataset-information)
3. [Machine Learning Models](#-machine-learning-models)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [How to Run the Pipeline](#-how-to-run-the-pipeline)
7. [Understanding the Outputs](#-understanding-the-outputs)
8. [Next Steps (Future Work)](#-next-steps-future-work)

---

## 🎯 Project Objectives

In the used car market, pricing is notorious for lacking transparency. This project utilizes Supervised Machine Learning algorithms to identify mathematical patterns linking a car's physical/historical traits to its ultimate valuation.
* **Stage 1 (Regression):** Predict the continuous `price` target variable using a baseline algorithm (Linear) and an improved distance-based algorithm (KNN).
* **Stage 2 (Classification):** Accurately classify the boolean `has_insurance` target variable (Future Implementation).

---

## 📊 Dataset Information
**File:** `data/pre-owned cars.csv`
**Size:** ~2,800 Rows
**Features (9):** `brand`, `model`, `make_year`, `fuel_type`, `transmission`, `engine_capacity(CC)`, `km_driven`, `ownership`, and `spare_key`.

*Note: The `overall_cost` feature was intentionally dropped during preprocessing to prevent Target Leakage.*

---

## 🧠 Machine Learning Models

### 1. Linear Regression (The Baseline)
A foundational statistical approach attempting to draw a "line of best fit" through the dataset. Because this dataset relies heavily on categorical strings (`brand`, `fuel_type`) encoded into integers, the linear equation struggles against the fake mathematical rankings, resulting in severe penalties and a negative $R^2$ score. 

### 2. K-Nearest Neighbours (KNN) Regression
The successful model. Instead of relying on a straight-line equation, KNN Regression groups cars based on feature similarity in multi-dimensional space. By averaging the prices of the 5 most mathematically similar vehicles, KNN successfully bypassed the linearity issues and achieved a highly accurate **0.77 $R^2$ Score**.

---

## 📂 Project Structure

```text
Supervised Learning Project/
│
├── data/
│   └── pre-owned cars.csv                # Raw dataset (Input)
│
├── docs/
│   ├── pkl_files_explained.md            # What are models/scalers and why they are saved
│   └── project_scope_and_interim.md      # Professor instructions and scope
│
├── outputs/                              # Automatically generated upon running
│   ├── scaler.pkl                        # Trained feature scaler (StandardScaler)
│   ├── linear_model.pkl                  # Trained baseline model
│   ├── knn_model.pkl                     # Trained KNN model
│   ├── linear_actual_vs_predicted.png    # Baseline regression scatter plot
│   └── knn_actual_vs_predicted.png       # KNN regression scatter plot
│
├── src/
│   ├── preprocessing.py                  # Cleans data, drops NaNs, LabelEncodes categories
│   ├── regression.py                     # Instantiates, trains, scales, and evaluates both models
│   └── classification.py                 # (Future Implementation)
│
├── main.py                               # The Root Orchestrator
├── presentation_content.md               # Script/notes for the Interim Presentation
├── .gitignore                            # Prevents pushing cached python files to Git
└── README.md                             # You are here
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   Download or clone this project folder to your local machine.
   
2. **Install Python:**
   Ensure you have Python 3.8+ installed on your system.

3. **Install Required Libraries:**
   Open your terminal/command prompt and run the following command to download the necessary data science libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

---

## 🚀 How to Run the Pipeline

The entire machine learning flow is automated. You do not need to run the `src/` modules individually.

1. Open your terminal.
2. Navigate (`cd`) into the root `Supervised Learning Project/` directory.
3. Run the main orchestrator script:
   ```bash
   python main.py
   ```

**What happens when you run `main.py`?**
* **Step 1:** The script locates the raw CSV file in `data/`.
* **Step 2:** It passes the data to `preprocessing.py` which cleans the NaNs, drops leaky columns, and label-encodes categorical data.
* **Step 3:** The cleaned features are pushed to `regression.py` which splits it 80/20.
* **Step 4:** Both Linear Regression and KNN are trained on the 80% split using `StandardScaler`.
* **Step 5:** Terminal outputs the absolute metrics (MAE, RMSE, $R^2$).
* **Step 6:** All massive `.pkl` models and high-resolution `.png` graphs are exported directly to the `outputs/` folder.

---

## 📈 Understanding the Outputs

After running the script, open your terminal to view the MAE and $R^2$ results.
Then, open your `outputs/` folder.

You will notice two `.png` files. 
* Look at `linear_actual_vs_predicted.png` to see a visual representation of a model struggling (dots scattered vertically away from the red prediction line).
* Look at `knn_actual_vs_predicted.png` to see the improvement. The dots cluster tightly along the diagonal red line, proving the model successfully learned to predict the vehicle prices based exclusively on neighboring similarities.

---

## 🔜 Next Steps (Future Work)

1. **Implement `classification.py`:** Predict the boolean `has_insurance` variable using Logistic Regression and KNN classifiers.
2. **Feature Importance Analysis:** Prove mathematically which columns impact the car price the most heavily.
3. **Hyperparameter Tuning:** Fine-tune the $k$ value in the KNN model to increase the $R^2$ accuracy above 80%.
4. **Decision Tree Regressors:** Introduce a third regression algorithm explicitly designed for categorical node-splitting.
