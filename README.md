# ❤️ Heart Disease Prediction Project  

This project focuses on predicting whether a patient has heart disease or not, based on various medical attributes. It leverages machine learning models with Python libraries like **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Scikit-Learn**.  

---

## 📌 Project Blueprint  

1. Problem Definition  
2. Data  
3. Evaluation  
4. Features  
5. Modeling  
6. Experimentation  

---

## 1. Problem Definition  
Given patient medical data, the goal is to **predict if a patient has heart disease (1) or not (0)**.  

---

## 2. Data  
Dataset: **heart.csv**  

- Contains patient medical information such as age, sex, cholesterol levels, heart rate, chest pain type, etc.  
- Target variable:  
  - `1` → Patient has heart disease  
  - `0` → Patient does not have heart disease  

---

## 3. Evaluation  
✅ Initial benchmark: Achieve **at least 95% accuracy** on predictions.  

We also evaluate using:  
- Confusion Matrix  
- Cross-Validation  
- Precision, Recall, F1-Score  
- ROC Curve  

---

## 4. Features  

| Feature | Description |
|---------|-------------|
| age     | Age in years |
| sex     | 1 = male, 0 = female |
| cp      | Chest pain type (0-3) |
| trestbps| Resting blood pressure (mm Hg) |
| chol    | Serum cholesterol (mg/dl) |
| fbs     | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang   | Exercise induced angina (1 = yes, 0 = no) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope   | Slope of the peak exercise ST segment |
| ca      | Number of major vessels (0–3) colored by fluoroscopy |
| thal    | 3 = normal, 6 = fixed defect, 7 = reversible defect |
| target  | 1 = disease, 0 = no disease |

---

## 5. Tools & Libraries  

- **Data Handling & Visualization**  
  - pandas, numpy, matplotlib, seaborn  

- **Machine Learning Models**  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Random Forest Classifier  

- **Evaluation & Tuning**  
  - train_test_split, cross_val_score  
  - RandomizedSearchCV, GridSearchCV  
  - classification_report, confusion_matrix  
  - precision_score, recall_score, f1_score, ROC curve  

---

## 6. Data Exploration  

- Checked for missing values, outliers, and data distribution  
- Visualized relationships between features and target variable (bar plots, scatter plots, histograms)  
- Generated correlation matrix and heatmaps to identify strong feature relationships  

Example Visualization:  
- **Age vs. Heart Rate** (Scatter Plot)  
- **Chest Pain vs. Heart Disease** (Bar Chart)  
- **Correlation Heatmap**  

---

## 7. Modeling  

- Split data into **training (80%)** and **testing (20%)** sets  
- Trained and compared multiple models:  
  - Logistic Regression  
  - Random Forest  
  - KNN  

- Created function to fit and score models on accuracy  

Example:  

```python
models = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}
