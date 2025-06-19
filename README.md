# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using supervised machine learning. Fraud detection is a critical application of data science, especially due to the class imbalance problem where fraudulent transactions are extremely rare compared to normal ones.

---

## 🧠 Objective

To build a binary classification model using **Logistic Regression** to accurately detect fraudulent credit card transactions.

---

## 📊 Dataset

The dataset used contains anonymized credit card transactions made by European cardholders in September 2013.

- Total Records: 284,807
- Fraudulent Transactions: 492 (≈0.172%)
- Features:
  - `V1` to `V28`: Principal components from PCA
  - `Amount`: Transaction amount
  - `Time`: Seconds from first transaction
  - `Class`: Target (0 = Legit, 1 = Fraud)

> 📌 Note: The dataset is highly imbalanced and was preprocessed to create a balanced version for training.

---

## ⚙️ Technologies Used

- Python 🐍
- NumPy & Pandas
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## 🛠️ Project Workflow

### 1. Data Preprocessing
- Loaded data using Pandas
- Explored dataset with `.info()`, `.describe()`, and `.value_counts()`
- Checked for missing values
- Observed significant class imbalance

### 2. Handling Imbalanced Data
- Used **random under-sampling**:
  - Took all 492 fraud cases
  - Sampled 492 legitimate cases randomly
  - Created a balanced dataset of 984 samples

### 3. Feature & Target Separation
- `X`: All features except 'Class'
- `Y`: Target variable (`Class`)

### 4. Train-Test Split
- Split into 80% training and 20% testing sets
- Used `stratify=Y` to maintain balance

### 5. Model Training
- Trained a **Logistic Regression** model on the balanced training data

### 6. Model Evaluation
- Evaluated using **accuracy score**
- Printed accuracy for both training and test sets

---

## ✅ Results

| Metric         | Score (example) |
|----------------|-----------------|
| Training Accuracy | ~95% |
| Test Accuracy     | ~93% |

> 🚨 Keep in mind that in real-world scenarios, accuracy may not be a sufficient metric for imbalanced classification tasks. Precision, recall, F1-score, and ROC-AUC are more informative.

---

## 🧠 Key Learnings

- How to handle imbalanced datasets using under-sampling
- Implementation of logistic regression for binary classification
- Importance of choosing the right evaluation metrics in fraud detection

---

## 📂 File Structure
📁 Credit_Card_Fraud_Detection/
│
├── credit_data.csv # Original dataset (not included due to size)
├── Project_10_Credit_Card_Fraud_Detection.ipynb # Jupyter Notebook
├── README.md # Project documentation
├── requirements.txt # Required dependencies
