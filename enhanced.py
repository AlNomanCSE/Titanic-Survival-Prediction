import pandas as pd
import warnings
from sklearn.preprocessing import(LabelEncoder,StandardScaler,MinMaxScaler) 
from sklearn.feature_selection import SelectKBest,f_classif
warnings.filterwarnings("ignore")

print("=" * 60)
print("🚢 ENHANCED TITANIC SURVIVAL PREDICTION PROJECT")
print("=" * 60)
try:
    df = pd.read_csv("train.csv")
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: Please download 'train.csv' from Kaggle Titanic competition")
    print("   Link: https://www.kaggle.com/c/titanic/data")
    exit()


df_enhanced = df.copy()

df_enhanced["FamilySize"] = df_enhanced["SibSp"] + df_enhanced["Parch"] + 1
df_enhanced["IsAlone"] = (df_enhanced["FamilySize"] == 1).astype(int)
df_enhanced["Title"] = df_enhanced["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

title_mapping = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Don": "Rare",
    "Rev": "Rare",
    "Dr": "Rare",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Major": "Rare",
    "Lady": "Rare",
    "Sir": "Rare",
    "Mlle": "Miss",
    "Col": "Rare",
    "Capt": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}
df_enhanced["Title"] = df_enhanced["Title"].map(title_mapping).fillna("Rare")

df_enhanced["AgeGroup"] = pd.cut(
    df_enhanced["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle", "Senior"],
)
df_enhanced["FareGroup"] = pd.qcut(
    df_enhanced["Fare"].fillna(df_enhanced["Fare"].median()),
    q=4,
    labels=["Low", "Medium", "High", "Very_High"],
)
df_clean = df_enhanced.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

for title in df_clean["Title"].unique():
    for pclass in df_clean["Pclass"].unique():
        mask = (df_clean["Title"] == title) & (df_clean["Pclass"] == pclass)
        age_median = df_clean[mask]["Age"].median()
        if not pd.isna(age_median):
            df_clean.loc[mask & df_clean["Age"].isna(), "Age"] = age_median

df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
df_clean['AgeGroup'].fillna('Adult', inplace=True)

# Remove duplicates
duplicates = df_clean.duplicated().sum()
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()

# 🔄 Enhanced Encoding
df_encoded = df_clean.copy()

# Label encode binary features
le_sex = LabelEncoder()
df_encoded["Sex_Encoded"] = le_sex.fit_transform(df_encoded["Sex"])

# One-hot encode categorical features
categorical_features = ['Embarked', 'Title', 'AgeGroup', 'FareGroup']
for feature in categorical_features:
    dummies = pd.get_dummies(df_encoded[feature], prefix=feature)
    df_encoded = pd.concat([df_encoded, dummies], axis=1)

# Drop original categorical columns
df_encoded.drop(columns=['Sex'] + categorical_features, inplace=True)

# 📏 Feature Scaling
df_scaled = df_encoded.copy()
numerical_columns = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
scaler = StandardScaler()
df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])

selector = SelectKBest(score_func=f_classif, k=15)
print(selector)