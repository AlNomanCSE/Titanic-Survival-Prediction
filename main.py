import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("🚢 TITANIC SURVIVAL PREDICTION PROJECT")
print("=" * 60)

# 📁 STEP 1: Loading and Exploring Data

try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("❌ Error: Please download 'train.csv' from Kaggle Titanic competition")
    print("   Link: https://www.kaggle.com/c/titanic/data")
    exit()


# 🧹 STEP 2: Data Cleaning
# Create a copy to preserve original data
df_clean = df.copy()

# Remove columns that won't help prediction
df_clean = df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
df_clean["Age"].fillna(df_clean["Age"].median(), inplace=True)
df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0], inplace=True)

# Check for duplicates (if then remove it)
duplicates = df_clean.duplicated().sum()
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()

# 🔄 STEP 3: Encoding Categorical Variables
df_encoded = df_clean.copy()
le_sex = LabelEncoder()
df_encoded["Sex_Encoded"] = le_sex.fit_transform(df_encoded["Sex"])

embarked_dummies = pd.get_dummies(df_encoded["Embarked"], prefix="Embarked")
df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
df_encoded.drop(columns=["Sex", "Embarked"], inplace=True)
# ✅ Encoding complete!


# Prepare for scaling
df_scaled = df_encoded.copy()
numerical_columns = ["Age", "SibSp", "Parch", "Fare"]
# scaler = StandardScaler()
scaler = MinMaxScaler()
df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
# print(df_scaled.columns)
# ✅ Applied StandardScaler to numerical features"


# 🎯 STEP 5: Preparing Features and Target
X = df_scaled.drop("Survived", axis=1)
y = df_scaled["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 🤖 STEP 6: Training Multiple Machine Learning Models
models = {}
results = {}

# 📊 Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
models["Logistic Regression"] = lr_model
# 🌳 Model 2: Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
models["Decision Tree"] = dt_model
# 👥 Model 3: K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
models["KNN"] = knn_model


# 📈 STEP 7: Model Evaluation
predictions = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "KNN": knn_pred,
}
# Evaluate each model
for modelName, pred in predictions.items():
    print(f"\n🔍 {modelName} Results:")
    print("-" * (len(modelName) + 10))
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # Store results
    results[modelName] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    # Display metrics
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    print(f"   Format: [[True Neg, False Pos], [False Neg, True Pos]]")

# 🏆 STEP 8: Model Comparison
print("=" * 60)
results_df = pd.DataFrame(results).T
print("📊 All Models Performance:")
print(results_df)

