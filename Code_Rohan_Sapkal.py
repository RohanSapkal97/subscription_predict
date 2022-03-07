import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

bank = pd.read_csv('BankMarketing.csv')
pd.set_option('display.max_columns', None)
print(bank.head())
print(bank.shape)
print(bank.info())
print(bank.describe())

#Replace 'unknown' error value to null
bank1 = bank.replace('unknown', np.NaN)

#Count of null value
bank1.isnull().sum()

#Drop all null value from dataset
bank2 = bank1.dropna(axis=0)

bank2.info()
bank2.isnull().sum()

#Mapping the variable to convert in integer
bank2['Credit']=bank2['Credit'].map({'no':0, 'yes':1})
bank2['Housing Loan']=bank2['Housing Loan'].map({'no':0, 'yes':1})
bank2['Personal Loan']=bank2['Personal Loan'].map({'no':0, 'yes':1})
bank2['Last Contact Month']=bank2['Last Contact Month'].map({'jan':0, 'feb':1, 'mar':2, 'apr':3, 'may':4, 'jun':5, 'jul':6, 'aug':7, 'sep':8, 'oct':9, 'nov':10, 'dec':11})
bank2['Contact']=bank2['Contact'].map({'cellular':0, 'telephone':1})

categorical_features = ['Job','Marital Status','Education','Poutcome']
final_data = pd.get_dummies(bank2, columns = categorical_features)

#To drop duplicate values
final_data.drop_duplicates()
print(final_data.shape)
print(final_data.info())

X = final_data.drop(['Subscription'], axis = 1) 
Y = final_data['Subscription']

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

#Implementing random forest with default parameter to create confussion matrix

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

# Building Classification Model
rfc = RandomForestClassifier()

rfc.fit(X_train, Y_train)
                             
# Evaluating random forest
Y_pred = rfc.predict(X_test)
print("Accuracy Score of Random forest:", metrics.accuracy_score(Y_test,Y_pred))
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("*Confusion Matrix*")
plt.xlabel("*Predicted Class*")
plt.ylabel("*Actual class*")
plt.show()
print('Confusion matrix for Random Forest: \n', conf_mat)
print('True Positive: ', conf_mat[1,1])
print('True Negative: ', conf_mat[0,0])
print('False Positive: ', conf_mat[0,1])
print('False Negative: ', conf_mat[1,0])

# Implementing Random Forest Classifier
# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1) )
    ])
grid_param = {'classification__n_estimators': [10,20,30,40,50,100]}

#Calling 'recall' score to minimize false negative as in confusion matrix FN needs to reduce. 
gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# Building random forest using the tuned parameter
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_scaled,Y)
featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp.head())

# Selecting features with higher significance and redefining feature set
X_ = final_data[['Last Contact Duration','Poutcome_success','Pdays','Balance (euros)']]

feature_scaler = StandardScaler()
X_scaled_ = feature_scaler.fit_transform(X_)

#Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 1)),
        ('classification', RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1) )
    ])
grid_param = {'classification__n_estimators': [10,20,30,40,50,100,150]}

#Calling 'recall' score to minimize false negative as in confusion matrix FN needs to reduce.
gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=5)

gd_sr.fit(X_scaled_, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

#Implementing SVM with default parameter to create confussion matrix

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

# Building Classification Model
svm = SVC()

svm.fit(X_train, Y_train)
                             
# Evaluating SVM
Y_pred = svm.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
print("Accuracy Score of SVM:", metrics.accuracy_score(Y_test,Y_pred))
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("*Confusion Matrix*")
plt.xlabel("*Predicted Class*")
plt.ylabel("*Actual class*")
plt.show()
print('Confusion matrix for SVM: \n', conf_mat)
print('True Positive: ', conf_mat[1,1])
print('True Negative: ', conf_mat[0,0])
print('False Positive: ', conf_mat[0,1])
print('False Negative: ', conf_mat[1,0])

# Implementing Support Vector Classifier
# Tuning the kernel parameter and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1) )
    ])
grid_param = {'classification__kernel': ['linear','poly','rbf','sigmoid'], 'classification__C': [.001,.01,.1,1,10,100]}

#Calling 'recall' score to minimize false negative as in confusion matrix FN needs to reduce.
gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)





