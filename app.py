import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
df=pd.read_csv('employee_attrition.csv')
drop_cols = ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y=(df['Attrition']=='Yes').astype(int)
num=X.select_dtypes(include=np.number).columns
cat=X.select_dtypes(exclude=np.number).columns
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),
                       ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),cat)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe=Pipeline([('preprocessor',pre),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))])
grid=GridSearchCV(pipe,{'model__C':[0.01,0.1,1,10]},cv=5,scoring='f1')
grid.fit(X_train,y_train)
pred=grid.predict(X_test)
print('Best parameters:',grid.best_params_)
print('Accuracy:',accuracy_score(y_test,pred))
print('F1:',f1_score(y_test,pred))
print(classification_report(y_test,pred))
joblib.dump(grid.best_estimator_,'models/employee_attrition_model.pkl')
