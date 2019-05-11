import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
pd.set_option('display.max_columns', 15)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.head())
print(df_train[df_train['Ticket'] == '3460'])

df_train.drop(['PassengerId', 'Ticket', 'Name'], axis = 1, inplace = True)


df_train = pd.get_dummies(df_train, columns = ['Sex'])
df_train.drop(['Sex_male'], axis = 1, inplace = True)

cabin = df_train['Cabin']
cabin.fillna('No', inplace = True)
cabin = cabin.values.tolist()
new_cabin = np.array([0 if x == 'No' else 1 for x in cabin]).T
df_train['Cabin'] = new_cabin

print(df_train[df_train['Embarked'].isna()])
df_train['Embarked'].fillna('S', inplace = True)
df_train['Embarked'].replace({'S' : 2, 'C' : 1, 'Q' : 0}, inplace = True)
#df_train = pd.get_dummies(df_train, columns = ['Embarked'])

df_train['Age'].fillna(value = np.random.randint(low = 16, high = 37), inplace = True)


mms = MinMaxScaler()
age = df_train['Age'].values.reshape(-1, 1)
age = mms.fit_transform(age)
df_train['Age'] = age

fare = df_train['Fare'].values.reshape(-1, 1)
fare = mms.fit_transform(fare)
df_train['Fare'] = fare

print(df_train.head())

y = df_train['Survived']
X = df_train.drop(['Survived'], axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 42)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)

print("Accuracy for Gradient Boosted Decision Trees : {}".format(accuracy_score(y_val, y_pred)))

passengers = df_test['PassengerId'].values
print(df_test[df_test['Ticket'] == '3702'])
print(df_test.iloc[152, :])
df_test.drop(['PassengerId', 'Ticket', 'Name'], axis = 1, inplace = True)

for column in df_test.columns.tolist():
    print("{} : {}".format(column, df_test[column].isna().sum()))

print(df_test[df_test['Fare'].isna()])

df_test = pd.get_dummies(df_test, columns = ['Sex'])
df_test.drop(['Sex_male'], axis = 1, inplace = True)


cabin = df_test['Cabin']
cabin.fillna('No', inplace = True)
cabin = cabin.values.tolist()
new_cabin = np.array([0 if x == 'No' else 1 for x in cabin]).T
df_test['Cabin'] = new_cabin

#df_test = pd.get_dummies(df_test, columns = ['Embarked'])
df_test['Embarked'].replace({'S' : 2, 'C' : 1, 'Q' : 0}, inplace = True)

df_test['Age'].fillna(value = np.random.randint(low = 16, high = 37), inplace = True)
df_test['Fare'].fillna(value = np.mean(df_test['Fare']), inplace = True)

mms = MinMaxScaler()
age = df_test['Age'].values.reshape(-1, 1)
age = mms.fit_transform(age)
df_test['Age'] = age

fare = df_test['Fare'].values.reshape(-1, 1)
fare = mms.fit_transform(fare)
df_test['Fare'] = fare

print(df_test.head())

y_pred = xgb.predict(df_test)

df = pd.DataFrame()
df['PassengerId'] = passengers.T
df['Survived'] = y_pred.T
df.to_csv('solution.csv', index = False)