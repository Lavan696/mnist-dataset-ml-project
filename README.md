#  MNIST Handwritten Digit Classification using K-Nearest Neighbors (KNN)

This project classifies handwritten digits from the **MNIST dataset** using a **K-Nearest Neighbors (KNN) classifier**.  
The dataset was fetched from **OpenML**, and the model was implemented using **Scikit-learn, NumPy, Pandas, Matplotlib, and Seaborn**.  

---

##  Project Overview  

The goal of this project is to accurately identify handwritten digits (0–9) using pixel intensity values.  
We trained a **KNN classifier**, optimized its hyperparameters, and performed **data augmentation** to improve the model’s robustness and accuracy.  

---

##  Steps Followed  

1. **Data Loading and Exploration**  
   - Loaded the MNIST dataset using `fetch_openml('mnist_784')`  
   - Extracted feature matrix `X` and label vector `y`  
   - Displayed dataset information and feature names  

2. **Visualization of Digits**  
   - Displayed a single digit and a grid of multiple digits using Matplotlib  

3. **Data Augmentation (Shifting Images)**  
   - Shifted images in **left, right, up, and down** directions by one pixel each  
   - Combined the original and shifted images to form an augmented dataset  
   - Resulting dataset size increased **5×**, improving generalization
   - This simple yet effective augmentation significantly improved model generalization by
     helping it learn **spatial invariance**—the ability to recognize digits even when slightly moved  

4. **Train-Test Split**  
   - Used `train_test_split` with a test ratio of **0.2**  

5. **Model Training (Before Hyperparameter Tuning)**  
   - Trained a basic **KNN classifier**  
   - Evaluated using **5-fold cross-validation**  

---

## Metrics Before Hyperparameter Tuning  

| Metric                      | Value  |
|-----------------------------|--------|
| Cross-Validation Mean Score | 96.87% |

---

##  Hyperparameter Tuning  

- Tuned model using `weights='distance'` and `n_neighbors=4`  
- Retrained the classifier on the augmented dataset  

---

##  Metrics After Hyperparameter Tuning  

| Metric                      | Value  |
|-----------------------------|--------|
| Cross-Validation Mean Score | 98.35% |
| Test Accuracy               | 98.59% |
| Precision (Weighted)        | 98.60% |
| Recall (Weighted)           | 98.59% |
| F1 Score (Weighted)         | 98.59% |
| ROC-AUC (OVR)               | 99.80% |
| Log Loss                    | 0.156  |
| Cohen Kappa Score           | 98.43% |
| Matthews Corr. Coefficient  | 98.43% |
| Top-2 Accuracy              | 99.62% |

---

##  Key Insights  

- **Data Augmentation** played a crucial role in improving accuracy and robustness.  
  By shifting images in multiple directions, the model learned to recognize digits even when they appear slightly displaced — a common trait in real-world handwriting.  
  This helped increase the **cross-validation mean score from 96.87% to 98.35%**, proving the power of data diversity.  
- **KNN with distance-based weighting** provided better performance compared to uniform weighting.  
- The **ROC-AUC score of 99.80%** indicates exceptional class separability.  
- The **Top-2 accuracy of 99.62%** suggests that even when the model misclassifies, the true label is almost always among its top predictions.  
- Overall, this approach demonstrates how simple preprocessing and augmentation can drastically improve traditional ML models without deep learning.  

---
##  Visualizations  

- **Sample Digit Images** – Displayed individual and grid-view digits  
- **Confusion Matrix** – Visualized classification performance   
- **Calibration Curve** – Showed predicted vs true probability alignment  
- **ROC Curve** – Multi-class ROC visualization with AUC for each digit  
- **Precision-Recall Curve** – Visualized tradeoff between precision and recall for each class  
- **Classification Report Heatmap** – Detailed view of per-class metrics  
- **Misclassified Digits** – Displayed 25 wrongly classified samples for insight  

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
## Model Saving  

The final trained model was saved using **Joblib** for easy reuse:  

`python
import joblib
joblib.dump(kn_best_clf, 'knn_best_model.pkl')
print('Model saved successfully!')`


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
