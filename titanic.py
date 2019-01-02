import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
data = pd.read_csv('/home/sarvesh/Titanic/train.csv')
pd.set_option('display.max_columns', None)
train = data
train.dropna(inplace = True)
train_target = train['Survived']
train.drop(['PassengerId', 'Name', 'Cabin', 'Survived', 'Embarked', 'Ticket'], axis = 1, inplace = True)
gender = {"male" : 0, "female" : 1}
train['Sex'].replace(gender, inplace = True)
train_columns = train
labels = list(train.columns.values)
labels.remove('Sex')
print(labels)
#for label in labels:
    #train_columns[label].fillna(value = train_columns[label].mean(), inplace = True)
print('Training Data :')
print(train_columns)
print(train_columns.describe())
test = pd.read_csv('/home/sarvesh/Titanic/test.csv')
passengers = test['PassengerId']
test.drop(['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket'], axis = 1, inplace = True)
test['Sex'].replace(gender, inplace = True)
test_columns = test
for label in labels:
    test_columns[label].fillna(value = test_columns[label].mean(), inplace = True)
print('Testing Data :')
print(test_columns)
print(test_columns.describe())
logreg = LogisticRegression(max_iter=8000)
svm = SVC()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' : ['l1', 'l2'], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
#'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#'loss' : ['hinge', 'squared_hinge']
#'penalty' : ['l1', 'l2']
#'kernel' : ['linear', 'rbf', 'poly', 'rbf', 'sigmoid', 'precomputed']
#svm.fit(train_columns, train_target)
#svm_clf = GridSearchCV(svm, param_grid)
#test_clf = GridSearchCV(logreg, param_grid)
#svm_clf.fit(train_columns, train_target)
clf = GridSearchCV(logreg, param_grid)
clf.fit(train_columns, train_target)
#svm_clf.fit(train_columns, train_target)
print(clf.best_params_)
#print(svm_clf.best_params_)
#clf.fit(train_columns, train_target)
#axis = 0 for rows
#axis = 1 for columns
#imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#steps = [('imputer', imp), ('gridsearchcv', clf)]
#pipeline = Pipeline(steps)
#pipeline.fit(train_columns, train_target)
#predictions = pipeline.predict(test)
#print("LinearSVC :")
#predictions = svm_clf.predict(test_columns) 
#print(predictions)
print("Logistic Regression :")
predictions = clf.predict(test_columns)
print(predictions)
#print(pipeline.named_steps['gridsearchcv'].score(train, train_target))
df = pd.DataFrame(passengers)
df['Survived'] = predictions
df.to_csv('/home/sarvesh/Titanic/solution.csv', index = False)