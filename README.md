# Day-72-Data-Processing

##  Overview
This repository is part of my **150 Days of Data & AI Journey**.

Today, I focused on **Data Processing**, which is a crucial step before building any Machine Learning model.

---

##  What is Data Processing?

Data Processing involves **cleaning, transforming, and organizing raw data** into a usable format for analysis and machine learning.

---

##  Key Concepts Covered

###  Data Cleaning
- Handling missing values
- Removing duplicates
- Fixing incorrect data

---

###  Data Transformation
- Normalization / Scaling
- Encoding categorical variables
- Feature selection

---

###  Data Preparation
- Splitting data (Train/Test)
- Structuring datasets for ML models

---

##  Tools Used

- Python 
- Pandas  
- NumPy  
- Scikit-learn  

---

##  Example Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    "Age": [25, 30, None, 35, 40],
    "Salary": [50000, 60000, 65000, None, 80000]
}

df = pd.DataFrame(data)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Features & target
X = df[["Age"]]
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Processed Data Ready for ML ")
