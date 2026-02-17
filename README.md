# Codveda Machine Learning Internship – Level 1 Projects

 **Hello and welcome!**  
This repository contains my solutions for the first two tasks of the Codveda Machine Learning internship.  
I worked with real‑world datasets to practice the fundamental steps of any machine learning project: **data preprocessing** and **building a simple predictive model**.

---

## 📌 Table of Contents
- [Datasets Used](#-datasets-used)
- [Task 1: Data Preprocessing](#-task-1-data-preprocessing)
- [Task 2: Linear Regression (House Price Prediction)](#-task-2-linear-regression-house-price-prediction)
- [How to Run the Code](#-how-to-run-the-code)
- [Key Takeaways](#-key-takeaways)
- [Connect with Me](#-connect-with-me)

---

## 📁 Datasets Used

| Task | Dataset | Description |
|------|---------|-------------|
| **Task 1** | `churn bigml.80.csv` | Customer churn data (80% sample). Contains customer demographics, account information, and whether they churned. |
| **Task 2** | `house_prices.csv` | House sales data with features like area, number of bedrooms, location, etc., and the sale price. |

Both files are included in this repository (or you can download them from [Kaggle / source link]).  
*Note: For Task 1, I used the 80% file as the raw dataset and split it further into training and test sets – exactly as the task required.*

---

## ✅ Task 1: Data Preprocessing for Machine Learning

**Goal:** Take a raw, messy dataset and clean it up so it’s ready for machine learning algorithms.

### What I Did

1. **Loaded the data** and explored its structure – checked column names, data types, missing values, and basic statistics.  
2. **Handled missing values**  
   - For numerical columns (like `Tenure`, `MonthlyCharges`), I filled empty cells with the **median** value.  
   - For categorical columns (like `Gender`, `PaymentMethod`), I filled missing values with the **most frequent** category.  
3. **Encoded categorical variables** – used **one‑hot encoding** to convert text categories into numbers that models can understand.  
4. **Scaled numerical features** – applied `StandardScaler` so all features have mean 0 and standard deviation 1. This helps many algorithms perform better.  
5. **Split the dataset** into a training set (80%) and a test set (20%) using `train_test_split`. The model will learn from the training set and be evaluated on the unseen test set.

### Files
- `Task1_Data_Preprocessing.ipynb` – the Jupyter notebook with all the code, explanations, and outputs.  
- `churn_clean.csv` (optional) – the cleaned dataset after preprocessing.

---

## ✅ Task 2: Linear Regression (House Price Prediction)

**Goal:** Build a linear regression model to predict house prices based on features like size, number of rooms, etc.

### What I Did

1. **Loaded the house prices dataset** and applied the same preprocessing steps from Task 1 (handle missing values, encode categories, scale numbers).  
2. **Split** into training and test sets (80/20).  
3. **Trained a Linear Regression model** using scikit‑learn – the model learns the best linear relationship between the features and the price.  
4. **Interpreted the coefficients** – each feature gets a coefficient that tells us its impact on price:  
   - Positive coefficient → as the feature increases, price tends to increase.  
   - Negative coefficient → as the feature increases, price tends to decrease.  
5. **Evaluated the model** with:  
   - **R² score** – measures how well the model explains the variance in prices (1 = perfect, 0 = no better than guessing).  
   - **Mean Squared Error (MSE)** – the average squared difference between predicted and actual prices (lower is better).  

### 📊 Visualizations

To better understand the model’s performance, I created two plots:

#### 1. Actual vs. Predicted Prices
This scatter plot compares the true prices with the model’s predictions. Ideally, points should hug the red diagonal line.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted House Prices')
plt.show()
```

#### 2. Residual Plot
Residuals (actual – predicted) should be randomly scattered around zero. Patterns would indicate the model is missing something.

```python
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

### Results

| Metric | Value |
|--------|-------|
| **R² Score** | 0.87 |
| **Mean Squared Error** | 1,250,000 |

*(These numbers are examples – replace them with your actual results!)*

### Files
- `Task2_Linear_Regression.ipynb` – the complete notebook with code, outputs, and plots.

---

## 🚀 How to Run the Code

1. **Clone this repository**  
   ```bash
   git clone https://github.com/jinxwycman/codveda-ml-level1.git
   cd codveda-ml-level1
   ```

2. **Install the required libraries** (preferably in a virtual environment)  
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

3. **Launch Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```

4. **Open the desired task notebook** (e.g., `Task1_Data_Preprocessing.ipynb`) and run the cells step by step.  
   Make sure the dataset files are in the same folder as the notebooks.

---

## Key Takeaways

- **Data preprocessing is the most time‑consuming but essential part** of any machine learning project. Clean data leads to better models.  
- **Linear regression is a simple yet powerful baseline** for regression problems. It’s also highly interpretable – you can see exactly how each feature influences the prediction.  
- **Always evaluate your model with multiple metrics and visualizations** – numbers alone can hide issues like non‑linear patterns or outliers.  
- **Splitting data properly** (train/test) prevents data leakage and gives a realistic estimate of model performance.


## 📬 Connect with Me

I’d love to hear your feedback or connect with you!

- **LinkedIn:** [Your Name](https://www.linkedin.com/in/machariajosepht/)  
- **GitHub:** [@yourusername](https://github.com/JinxWycman)  
- **Email:** machariajoseph1422@gmail.com


**#CodvedaJourney #CodvedaExperience #MachineLearning #DataScience #Python**
