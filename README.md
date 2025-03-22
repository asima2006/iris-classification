# Iris Flower Classification

## 📌 Overview
This project demonstrates how to classify iris flowers using machine learning. The dataset used is the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) from scikit-learn, which contains 150 samples across three species of iris flowers:
- **Setosa**
- **Versicolor**
- **Virginica**

Each sample has four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The goal is to train a model that can accurately classify iris flowers based on these features.

## 📂 Project Structure
```
Iris_Flower_Classification/
├── Iris_Flower_Classification_git.ipynb   # Jupyter Notebook with the full project code
├── README.md                              # Project documentation
└── requirements.txt                        # List of required dependencies
```

## 💻 Requirements
- **Python 3.11**
- **Jupyter Notebook**

### Required Python Packages
You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can install them individually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/asima2006/iris-classification.git
   cd Iris_Flower_Classification
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Iris_Flower_Classification_git.ipynb
   ```

3. **Run the Notebook Cells:**
   Execute each cell in order to:
   - Load and explore the dataset.
   - Preprocess the data.
   - Train and evaluate machine learning models.
   - Visualize results.

## 📜 Code Explanation

### 1️⃣ Data Loading
The Iris dataset is loaded from scikit-learn:
```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
```

### 2️⃣ Data Exploration
- Understanding dataset properties using `pandas`.
- Visualizing data using `seaborn` and `matplotlib`.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()
```

### 3️⃣ Data Preprocessing
Splitting the dataset into training and test sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4️⃣ Model Training
Training a Logistic Regression model:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

### 5️⃣ Model Evaluation
Evaluating the model using accuracy and confusion matrix:
```python
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## 📊 Results
- The model achieves a high accuracy in classifying iris species.
- Visualization helps in understanding feature importance and classification performance.

## 🛠 Future Improvements
- Experiment with different classification models (e.g., Decision Trees, Random Forests, SVMs).
- Tune hyperparameters for improved accuracy.
- Deploy the model as a web application using Flask or FastAPI.

## 🤝 Acknowledgments
- **scikit-learn** for providing the Iris dataset.
- The open-source community for valuable machine learning resources.

---
### ⭐ If you found this project helpful, give it a star! ⭐