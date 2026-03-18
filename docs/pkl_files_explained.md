# What are .pkl Files?

## What does `.pkl` mean?

`.pkl` stands for **Pickle**. It is a Python file format used to **save (serialize) Python objects** to disk, so they can be reloaded later without needing to recreate them from scratch.

Think of it like this: instead of re-training a model or refitting a scaler every time you run the program, you save it once into a `.pkl` file and simply load it whenever needed.

This is done using the **`joblib`** library, which is efficient for saving large objects like those used by scikit-learn models.

---

## PKL Files in This Project

### 1. `scaler.pkl`

| Property | Detail |
|---|---|
| **Created by** | `StandardScaler` from `sklearn.preprocessing` |
| **What it contains** | The calculated mean and standard deviation of every feature in the training dataset |
| **Why it is saved** | Because both **Linear Regression** and **KNN Regression** are highly sensitive to the scale of numbers. The scaler ensures that large numbers (like `km_driven`) don't overpower smaller numbers (like `engine_capacity`). Any new car data must be scaled using these exact same historical metrics before predicting a price. |

### 2. `linear_model.pkl`

| Property | Detail |
|---|---|
| **Created by** | `LinearRegression` from `sklearn.linear_model` |
| **What it contains** | The fully trained Linear Regression model — specifically the optimal **intercept** ($\beta_0$) and feature **coefficients** ($\beta_1, \beta_2$, etc.) it learned from the historical data. |
| **Why we use it** | Linear Regression is the primary, foundational regression model required for the interim presentation phase of this course. |

### 3. `knn_model.pkl`

| Property | Detail |
|---|---|
| **Created by** | `KNeighborsRegressor` from `sklearn.neighbors` |
| **What it contains** | The trained KNN model, which memorizes the optimized positioning of all historical car datapoints. It does not learn an equation; rather, it learns how to rapidly calculate distances to the $k$ nearest cars when a new prediction is requested. |
| **Why we use it** | After calculating that Linear Regression fails on this categorical dataset, we introduced KNN as an advanced improvement strategy to successfully predict prices based on feature "similarity" rather than drawing a straight equation line. |

---

## Summary

| File | Contains | Used for |
|---|---|---|
| `scaler.pkl` | Training mean & standard deviation | Ensuring new data is mathematically scaled correctly before testing |
| `linear_model.pkl` | Trained Linear Regression formulas | Predicting continuous car prices ($y$) using equation coefficients |
| `knn_model.pkl` | Spatial dataset representation | Predicting car prices ($y$) by averaging the 5 most similar cars |

Saved to the `outputs/` folder in this project.
