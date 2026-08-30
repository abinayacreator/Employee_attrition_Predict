# Employee Attrition Prediction

## 1. Problem Statement

Employee attrition is a major challenge faced by organizations across different industries. When employees leave an organization, it can result in increased recruitment and training costs, loss of experienced employees, reduced productivity, and disruption to team performance.

The objective of this project is to develop a Machine Learning-based Employee Attrition Prediction system that can predict whether an employee is likely to leave the organization based on various employee-related factors.

The dataset contains information such as Age, Business Travel, Department, Distance From Home, Education, Environment Satisfaction, Job Involvement, Job Level, Job Role, Job Satisfaction, Monthly Income, OverTime, Total Working Years, Work-Life Balance, Years at Company, and other relevant attributes.

This project treats employee attrition prediction as a binary classification problem, where the target variable is `Attrition`.

The Machine Learning model analyzes historical employee data and identifies patterns associated with employee attrition. The final model can help organizations identify employees who may be at higher risk of leaving and support data-driven employee retention strategies.

---

## 2. Objectives

- To analyze employee-related data and understand attrition patterns.
- To identify factors associated with employee attrition.
- To perform data cleaning and preprocessing.
- To handle missing values and duplicate records.
- To identify numerical and categorical features.
- To encode categorical variables.
- To perform Exploratory Data Analysis (EDA).
- To create meaningful visualizations.
- To build Machine Learning classification models.
- To compare different Machine Learning algorithms.
- To select a suitable final model.
- To improve model performance using hyperparameter tuning.
- To evaluate the final model using suitable metrics.
- To save the trained Machine Learning model.
- To develop a Streamlit-based prediction application.

---

## 3. Dataset Description

The dataset contains employee information that can be used to understand and predict employee attrition.

**Problem Type:** Binary Classification

**Target Variable:** `Attrition`

The target variable contains two possible outcomes:

- `Yes` - Employee attrition
- `No` - Employee did not leave

### Feature Dictionary

| Feature | Description |
|---|---|
| Age | Age of the employee |
| BusinessTravel | Frequency of business travel |
| DailyRate | Employee's daily rate |
| Department | Department in which the employee works |
| DistanceFromHome | Distance between home and workplace |
| Education | Education level |
| EducationField | Field of education |
| EnvironmentSatisfaction | Satisfaction with work environment |
| Gender | Gender of the employee |
| JobInvolvement | Level of job involvement |
| JobLevel | Employee's job level |
| JobRole | Employee's job role |
| JobSatisfaction | Employee's job satisfaction |
| MaritalStatus | Marital status |
| MonthlyIncome | Monthly income |
| NumCompaniesWorked | Number of companies previously worked for |
| OverTime | Whether the employee works overtime |
| PercentSalaryHike | Percentage salary increase |
| PerformanceRating | Employee performance rating |
| RelationshipSatisfaction | Workplace relationship satisfaction |
| StockOptionLevel | Employee stock option level |
| TotalWorkingYears | Total years of work experience |
| TrainingTimesLastYear | Training sessions attended |
| WorkLifeBalance | Work-life balance rating |
| YearsAtCompany | Years spent at the company |
| YearsInCurrentRole | Years spent in current role |
| YearsSinceLastPromotion | Years since last promotion |
| YearsWithCurrManager | Years working with current manager |
| Attrition | Target variable |

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Google Colab
- GitHub

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the employee dataset using Pandas.
2. Examined the dataset structure and data types.
3. Checked for missing values.
4. Checked for duplicate records.
5. Removed duplicate records where required.
6. Identified numerical and categorical features.
7. Handled missing values appropriately.
8. Encoded categorical variables.
9. Separated input features and target variable.
10. Prepared the dataset for Machine Learning.
11. Split the dataset into training and testing sets.

### Train-Test Split

- Training Data: 80%
- Testing Data: 20%
- Random State: 42

---

## 6. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand employee characteristics and identify patterns related to attrition.

### EDA Performed

- Attrition distribution
- Age distribution
- Monthly income analysis
- Job satisfaction analysis
- Overtime analysis
- Job role analysis
- Department-wise attrition analysis
- Distance from home analysis
- Years at company analysis
- Work-life balance analysis
- Correlation analysis
- Outlier analysis

### Visualizations

The project includes:

- Count plots
- Bar charts
- Histograms
- Box plots
- Correlation heatmaps
- Attrition distribution charts
- Feature relationship plots

---

## 7. Machine Learning Models

Since `Attrition` is a binary target variable, classification algorithms were used.

The following Machine Learning models were evaluated:

1. Logistic Regression
2. Gradient Boosting Classifier
3. Random Forest Classifier

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.17% | 34.88% | 63.83% | 45.11% | 80.32% |
| Gradient Boosting | 86.05% | 71.43% | 21.28% | 32.79% | 80.51% |
| Random Forest | 84.69% | 62.50% | 10.64% | 18.18% | 79.03% |

---

## 8. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Although Gradient Boosting achieved the highest accuracy of **86.05%**, its recall was comparatively low.

For employee attrition prediction, identifying employees who may leave is important. Therefore, Recall is an important evaluation metric because missing a potential attrition case can reduce the usefulness of the prediction system.

Logistic Regression achieved a higher recall of **63.83%** compared with the other evaluated models.

---

## 9. Model Improvement

The Logistic Regression model was further optimized using hyperparameter tuning.

### Before and After Tuning

| Metric | Before Tuning | After Tuning |
|---|---:|---:|
| Accuracy | 75.17% | 77.21% |
| Precision | 34.88% | 38.10% |
| Recall | 63.83% | 68.09% |
| F1-Score | 45.11% | 48.85% |
| ROC-AUC | 80.32% | 81.24% |

The tuned model showed improvements in accuracy, precision, recall, F1-score, and ROC-AUC.

### Final Model

**Final Model: Tuned Logistic Regression**

The tuned Logistic Regression model was selected as the final model because employee attrition prediction requires identifying employees who are likely to leave. Therefore, recall is an important metric.

---

## 10. Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 77.21% |
| Precision | 38.10% |
| Recall | 68.09% |
| F1-Score | 48.85% |
| ROC-AUC | 81.24% |

The ROC-AUC score of **81.24%** indicates that the final model has good capability to distinguish between employees with and without attrition.

---

## 11. Streamlit Application

A Streamlit web application was developed to provide an interactive Employee Attrition Prediction system.

### Application Workflow

Employee Details  
↓  
Input Validation  
↓  
Data Preprocessing  
↓  
Trained ML Model  
↓  
Attrition Prediction  
↓  
Prediction Result

The application allows users to provide employee-related information and obtain a predicted attrition result from the trained model.

---

## 12. Project Structure

```text
Employee_Attrition_Prediction/
│
├── data/
│   └── employee_attrition.csv
│
├── models/
│   └── employee_attrition_model.pkl
│
├── notebooks/
│   └── Employee_Attrition_Analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── reports/
│   └── model_comparison.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
13. Installation
Clone the repository:
git clone <YOUR_GITHUB_REPOSITORY_URL>
Navigate to the project folder:
cd Employee_Attrition_Prediction
Install the required packages:
pip install -r requirements.txt
14. Run the Application
Run the Streamlit application using:
streamlit run app.py
After running the command, open the local Streamlit URL displayed in the terminal.
15. Applications
This Employee Attrition Prediction system can be useful for:
Employee retention analysis
HR analytics
Workforce planning
Identifying potential attrition cases
Supporting data-driven HR decisions
Understanding factors associated with employee turnover
16. Limitations
The prediction model is based on historical employee data. Therefore, predictions should be treated as decision-support information rather than definitive conclusions about an individual employee.
Model performance may also change when applied to employee populations with different characteristics from the training dataset.
17. Future Enhancements
Future improvements may include:
Trying additional Machine Learning algorithms.
Advanced feature engineering.
Handling class imbalance using appropriate techniques.
Advanced hyperparameter optimization.
Explainable AI techniques such as SHAP.
Improved Streamlit dashboard.
Real-time prediction.
Cloud deployment.
Model monitoring and periodic retraining.
