#  MNIST Handwritten Digit Classification using K-Nearest Neighbors (KNN)  

This project classifies handwritten digits from the **MNIST dataset** using a **K-Nearest Neighbors (KNN) classifier**.  
The dataset was taken from **OpenML**, and the model was implemented using **Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn**.  

---

##  Project Overview  

The goal of this project is to correctly identify digits (0–9) from images of handwritten numbers.  
Trained a **KNN classifier**, evaluated it, and optimized its hyperparameters to achieve high accuracy and robust performance metrics.  

---

##  Steps Followed  

1. **Data Loading and Exploration**  
   - Loaded MNIST dataset from `fetch_openml`  
   - Explored **features** (`X`) and **labels** (`y`)  
   - Displayed value counts and feature names  

2. **Visualization of Digit Images**  
   - Displayed a single digit  
   - Displayed multiple sample images (first 100 digits)  

3. **Train-Test Split**  
   - Split dataset into **training set (60,000 samples)** and **test set (10,000 samples)**  

4. **Model Training (Before Hyperparameter Tuning)**  
   - Trained **default KNN classifier**  
   - Evaluated using **10-fold cross-validation**  

5. **Metrics Before Hyperparameter Tuning**  

| Metric                      | Value  |
|-----------------------------|--------|
| Cross-Validation Mean Score | 97.00% |
| Test Accuracy               | 96.88% |
| Precision (Weighted)        | 96.90% |
| Recall (Weighted)           | 96.88% |
| F1 Score (Weighted)         | 96.87% |

6. **Hyperparameter Tuning**  
   - Optimized KNN with `weights='distance'`, `n_neighbors=4`, and `n_jobs=-1`  
   - Retrained the model on the training set  

7. **Metrics After Hyperparameter Tuning**  

| Metric                      | Value  |
|-----------------------------|--------|
| Cross-Validation Mean Score | 97.28% |
| Test Accuracy               | 97.14% |
| Precision (Weighted)        | 97.15% |
| Recall (Weighted)           | 97.14% |
| F1 Score (Weighted)         | 97.13% |
| ROC-AUC Score               | 99.43% |
| Log Loss                    | 0.404  |
| Cohen Kappa Score           | 96.82% |
| Matthews Corr. Coefficient  | 96.82% |
| Top-3 Accuracy              | 99.15% |

---

## Visualizations  

- **Sample Digit Images** – Displayed single and multiple digits  
- **Confusion Matrix** – Before and after hyperparameter tuning  
- **Calibration Curve** – Predicted vs true probability  
- **ROC Curve** – Multi-class ROC for each digit  
- **Precision-Recall Curve** – Multi-class precision-recall curves  
- **Classification Report Heatmap** – Weighted metrics per class  
- **Misclassified Digits** – Visual display of 25 misclassified images  

---

## Key Insights  

- Hyperparameter tuning improved accuracy, precision, recall, and F1-score slightly but consistently.  
- ROC-AUC of 99.43% indicates excellent separability between digit classes.  
- Top-3 accuracy of 99.15% shows that in most cases, the true digit is among the top 3 predicted classes.  
- Confusion matrix and misclassified images provide insight into which digits are most commonly confused.  

---

##  Tech Stack  

- **Programming Language:** Python 
- **Libraries:**  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Seaborn  

---
##  Saved Model

The trained KNN model is saved using **Joblib**:

`python
import joblib
knn_model = joblib.load('models/knn_mnist_model.pkl')
y_pred = knn_model.predict(X_test)`

##  Author  

**Lavan kumar Konda**  
-  Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
- 🔗 [LinkedIn](www.linkedin.com/in/lavan-kumar-konda)
  
---
##  How to Run  

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/mnist-knn-project.git
