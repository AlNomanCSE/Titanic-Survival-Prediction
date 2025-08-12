import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("🚢 TITANIC SURVIVAL PREDICTION PROJECT")
print("=" * 60)

print("\n📁 STEP 1: Loading and Exploring Data")
print("-" * 40)

try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("❌ Error: Please download 'train.csv' from Kaggle Titanic competition")
    print("   Link: https://www.kaggle.com/c/titanic/data")
    exit()

print("\n🎯 See target variable distribution:")
print(df["Survived"].value_counts())

print("\n🧹 STEP 2: Data Cleaning")
print("-" * 40)
print(df.isnull().sum())
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
    

