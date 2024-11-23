# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the `pandas` library and read the dataset `Placement_Data.csv` into a DataFrame `data`.
2. Display the first few rows of `data` using `data.head()` to check its contents.
3. Create a copy of `data` named `data1` to work on.
4. Drop the columns `"sl_no"` and `"salary"` from `data1` using `data1.drop()`.
5. Display the first few rows of `data1` after dropping columns to confirm changes.
6. Check for missing values in `data1` using `data1.isnull().sum()`.
7. Check for duplicate rows in `data1` using `data1.duplicated().sum()`.
8. Import `LabelEncoder` from `sklearn.preprocessing` for encoding categorical variables.
9. Create an instance of `LabelEncoder` named `le`.
10. Encode categorical columns (`"gender"`, `"ssc_b"`, `"hsc_b"`, `"hsc_s"`, `"degree_t"`, `"workex"`, `"specialisation"`, `"status"`) in `data1` using `le.fit_transform()`.
11. Display `data1` after encoding to check that categorical columns are encoded into numerical values.
12. Extract the independent variables `x` (all columns except `"status"`) from `data1`.
13. Extract the dependent variable `y` (the `"status"` column) from `data1`.
14. Split `x` and `y` into training and testing sets using `train_test_split()` with 80% training and 20% testing data, setting `random_state` to 0 for reproducibility.
15. Import `LogisticRegression` from `sklearn.linear_model`.
16. Create an instance of `LogisticRegression` named `lr` with the solver set to `"liblinear"`.
17. Train the model on the training data using `lr.fit()` with `x_train` and `y_train`.
18. Make predictions on the test set `x_test` using `lr.predict()` and store them in `y_pred`.
19. Print or return `y_pred` to display the predicted output. 
## Program:
### 1.PLACEMENT DATA:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LINGESWARAN K
RegisterNumber:  212222110022
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
### 2.SALARY DATA:
```
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1
```
### 3.PRINT DATA:
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
### 4.DATA STATUS:
```
x=data1.iloc[:, : -1]
x
```
```
y=data1["status"]
y
```
### 5.Y_PREDICATION ARRAY: 
```

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
```
### 6.ACCURACY SCORE:
### 7.CONFUSION MATRIX:
### 8.Classification Report::
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
```
### 9.CONFUSION MATRIX IN DISPLAY FORMAT:
```
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:
### 1.PLACEMENT DATA:
![image](https://github.com/user-attachments/assets/169ee3ca-6813-4b0a-a91c-7e0f2fafcbb8)
### 2.SALARY DATA:
![image](https://github.com/user-attachments/assets/81e1b7b0-0b83-439c-9ea2-d562ebc9cde4)
### 3.PRINT DATA:
![image](https://github.com/user-attachments/assets/66d40b61-39b4-4bb7-834f-731f0e9448d5)
### 4.DATA STATUS:
![image](https://github.com/user-attachments/assets/49143666-a324-4e9b-abb3-4f882b5ff94f)
### 5.Y_PREDICATION ARRAY: 
![image](https://github.com/user-attachments/assets/f6430e63-93c4-4d6a-85fc-bd21e8680268)

![image](https://github.com/user-attachments/assets/db1c601e-2f05-456c-bfe1-49b175dc8148)
### 6.ACCURACY SCORE:
![image](https://github.com/user-attachments/assets/620b8639-3d08-49ba-b791-65d4f444e005)
### 7.CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/2fb0ee4d-4532-4050-907b-e929f0f46fdd)
### 8.Classification Report:
![image](https://github.com/user-attachments/assets/c98fd135-96d0-4364-800e-f7f39ce2b81b)
### 9.CONFUSION MATRIX IN DISPLAY FORMAT:
![image](https://github.com/user-attachments/assets/99d21304-5354-476e-983e-c954b8e84d36)

## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
