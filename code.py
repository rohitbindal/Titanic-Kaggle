# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")
df.drop([
    "Name",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin"
],axis = 1,inplace = True)

# Updating Null values with the Medians
df["Age"].fillna(df.Age.median(),inplace=True)
df["Embarked"].fillna(df.Embarked.value_counts().index[0],inplace=True)
# df = sklearn.utils.shuffle(df)
df.set_index("PassengerId")

x_train = df[:600]
y_train = df[:600]
x_test = df[600:]
y_test = df[600:]


x_train.drop("Survived",axis=1,inplace=True)
x_test.drop("Survived",axis=1,inplace=True)
y_train.drop(["Pclass","Sex","Age","Embarked"],axis=1,inplace=True)
y_test.drop(["Pclass","Sex","Age","Embarked"],axis=1,inplace=True)

x_train = x_train.set_index("PassengerId")
y_train = y_train.set_index("PassengerId")
x_test = x_test.set_index("PassengerId")
y_test = y_test.set_index("PassengerId")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x_train['Sex'] = le.fit_transform(x_train['Sex'])
x_train['Pclass'] = le.fit_transform(x_train['Pclass'])
x_train['Embarked'] = le.fit_transform(x_train['Embarked'])
x_test['Sex'] = le.fit_transform(x_test['Sex'])
x_test['Pclass'] = le.fit_transform(x_test['Pclass'])
x_test['Embarked'] = le.fit_transform(x_test['Embarked'])

# Converting dataframes to numpy array
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values

# Using Logistic Regression to predict the survival of an individual

# Sigmoid function to predict the probability of survival
def sigmoid(features, parameters):
    z = np.dot(features, parameters.transpose())
    return (1/(1+np.exp(-z)))   #returns a 1-D array of Probablities

def cost_function(features, labels, parameters):
    m = len(labels)  #Number of training examples
    h = sigmoid(features, parameters)  #Hypothesis
    #when y=1
    h[h==1]=0.999
    cost_1 = np.dot(np.transpose(labels),np.log(h))
    #when y = 0
    cost_2 = np.dot(np.transpose(1-labels),np.log(1-h))
    
    final_cost = -(cost_1 + cost_2)
    final_cost = final_cost/m
    
    return final_cost  #returns a 1-D matrix of predictions

def gradient_descent(features, labels, parameters, lr):
    m = len(features)  #number of training examples
    h = sigmoid(features, parameters)
    
    gradient = np.dot(features.T,h-labels)
    gradient = gradient/m
    gradient = gradient*lr;
    
    parameters = parameters-gradient.transpose()
    return parameters

def train(features, labels, parameter, lr, iterations):
    cost_history=[]
    
    for i in range(iterations):
        parameter = gradient_descent(features, labels, parameter, lr)
        cost = cost_function(features, labels, parameter)
        cost_history.append(cost)
        if i%1000==0:
            print("Iteration:"+str(i)+" Cost:"+str(cost))
    
    return parameter, cost_history

lr = 0.0031
parameters = np.random.rand(1,4)
prediction = train(x_train,y_train,parameters,lr,50000)

def predict(features, parameters):
    h = sigmoid(features,parameters)
    return h

def accuracy(predictions,labels):
    print("Prediction Accuracy: ",str(np.mean(predictions==labels)*100)+"%")

theta = prediction[0]
val = predict(x_train,theta)

# Change the values to 0 and 1
val[val>=0.5]=1
val[val<0.5]=0
accuracy(val,y_train)


# Let's use the test data
test = predict(x_test,theta)
test[test>=0.5]=1
test[test<0.5]=0
accuracy(test,y_test)

df_test = pd.read_csv("../input/test.csv")
passengerId = df_test["PassengerId"]
df_test.drop([
    "PassengerId",
    "Name",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin"
],axis = 1,inplace = True)

# Updating Null values with the Medians
df_test["Age"].fillna(df_test.Age.median(),inplace=True)
df_test['Sex'] = le.fit_transform(df_test['Sex'])
df_test['Pclass'] = le.fit_transform(df_test['Pclass'])
df_test['Embarked'] = le.fit_transform(df_test['Embarked'])

# Converting dataframes to numpy array
df_test = df_test.values
# Predict the survivality of an Individual
final_test = predict(df_test,theta)
final_test[final_test>=0.5]=1
final_test[final_test<0.5]=0

sub = np.insert(final_test,0,passengerId,axis=1)
sub = pd.DataFrame({'PassengerId':sub[:,0],'Survived':sub[:,1]})
sub = sub.astype(int)
sub.to_csv('submission.csv',index=False)

