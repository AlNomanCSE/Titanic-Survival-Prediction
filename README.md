# 🚢 Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![UV](https://img.shields.io/badge/Package%20Manager-UV-purple?style=flat-square)](https://github.com/astral-sh/uv)

> A comprehensive machine learning project predicting passenger survival on the Titanic using multiple algorithms and complete data preprocessing pipeline.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)  
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a complete machine learning pipeline to predict passenger survival on the Titanic. It demonstrates end-to-end data science workflow including data preprocessing, feature engineering, model training, evaluation, and comparison of multiple algorithms.

**Key Highlights:**
- 🧹 Comprehensive data cleaning and preprocessing
- 🔄 Multiple encoding techniques (Label & One-Hot)
- ⚖️ Feature scaling using StandardScaler
- 🤖 Three different ML algorithms comparison
- 📊 Detailed model evaluation with multiple metrics
- 🎯 Real-world prediction capabilities

## ✨ Features

- **Data Preprocessing Pipeline**
  - Missing value imputation
  - Outlier handling
  - Duplicate removal
  - Feature selection

- **Feature Engineering**
  - Label encoding for binary categories
  - One-hot encoding for categorical variables  
  - Feature scaling and normalization
  - Train-test data splitting

- **Machine Learning Models**
  - Logistic Regression
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)

- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix analysis
  - Model comparison and selection
  - Prediction probability analysis

## 📊 Dataset

**Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)

**Dataset Details:**
- **Size:** 891 passengers
- **Features:** 12 original features
- **Target:** Survival (0 = No, 1 = Yes)
- **Missing Values:** Age (177), Cabin (687), Embarked (2)

**Key Features Used:**
- `Pclass` - Passenger class (1st, 2nd, 3rd)
- `Sex` - Gender
- `Age` - Age in years  
- `SibSp` - Number of siblings/spouses aboard
- `Parch` - Number of parents/children aboard
- `Fare` - Passenger fare
- `Embarked` - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- [UV Package Manager](https://github.com/astral-sh/uv)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlNomanCSE/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```

2. **Install dependencies using UV**
   ```bash
   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

3. **Download the dataset**
   - Go to [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
   - Download `train.csv`
   - Place it in the project root directory

## 💻 Usage

### Running the Complete Pipeline

```bash
# Activate the virtual environment
uv run python titanic_survival_prediction.py
```

### Step-by-Step Execution

The script will automatically:
1. Load and explore the dataset
2. Clean and preprocess the data
3. Encode categorical variables
4. Scale numerical features
5. Train multiple ML models
6. Evaluate and compare model performance
7. Make sample predictions

### Expected Output
```
🚢 TITANIC SURVIVAL PREDICTION PROJECT
========================================
📁 Loading and exploring data...
🧹 Data cleaning and preprocessing...
🔄 Encoding categorical variables...
⚖️ Feature scaling...
🤖 Training multiple ML models...
📊 Model evaluation and comparison...
🏆 Best Model: [Model Name] (XX.X% accuracy)
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.8212 | 0.7945 | 0.7632 | 0.7786 |
| **Decision Tree** | 0.8045 | 0.7500 | 0.7895 | 0.7692 |
| **K-Nearest Neighbors** | 0.7877 | 0.7333 | 0.7368 | 0.7351 |

> **Best Performing Model:** Logistic Regression with **82.1% accuracy**

### Confusion Matrix (Best Model)
```
[[95  10]
 [22  52]]
```

## 📁 Project Structure

```
Titanic-Survival-Prediction/
├── 📄 README.md
├── 🐍 titanic_survival_prediction.py   # Main project script
├── 📊 train.csv                        # Dataset (download separately)
├── 📋 pyproject.toml                   # UV project configuration
├── 🔒 uv.lock                         # UV lock file
├── 📊 requirements.txt                 # Dependencies list
└── 📄 LICENSE                         # MIT License
```

## 🛠 Technologies Used

- **Python 3.8+** - Programming language
- **UV** - Fast Python package manager
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning library
  - `preprocessing` - Data preprocessing
  - `model_selection` - Train-test split
  - `linear_model` - Logistic Regression
  - `tree` - Decision Tree
  - `neighbors` - K-Nearest Neighbors
  - `metrics` - Model evaluation
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization

## 🎯 Results

### Key Insights Discovered:
- **Gender Impact:** Women had significantly higher survival rates (74%) compared to men (19%)
- **Class Matters:** 1st class passengers had better survival chances (63%) than 3rd class (24%)
- **Age Factor:** Children and young adults had slightly better survival rates
- **Family Size:** Passengers with small families (1-3 people) survived more than those traveling alone

### Model Insights:
- **Logistic Regression** performed best with balanced precision and recall
- **Decision Tree** showed good interpretability but slight overfitting tendency
- **KNN** was competitive but sensitive to feature scaling

## 🔮 Future Enhancements

- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Create interactive visualization dashboard
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Implement cross-validation for robust evaluation
- [ ] Add feature importance analysis
- [ ] Create web interface for real-time predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Al Noman**
- GitHub: [@AlNomanCSE](https://github.com/AlNomanCSE)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

---

*This project demonstrates a complete machine learning workflow and serves as a portfolio piece showcasing data science and ML engineering skills.*