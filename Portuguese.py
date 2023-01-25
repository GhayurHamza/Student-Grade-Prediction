###Student's final grade prediction using python and machine learning###
     # This Python 3 environment comes with many helpful analytics libraries installed
     # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
     # For example, here's several helpful packages to load

     #This code is for importing two important libraries numpy and pandas for importing and manipulating
     #data. np is numerical python and pd is data manipulation library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
###performed by Ghayur Hamza_19221_ECE###

##DATA INPUT
data=pd.read_csv("/kaggle/input/d/datasets/larsen0966/student-performance-data-set/student-por.csv")
data.head()

#data visualization using matplotlib and seaborn
import seaborn as sns 
import matplotlib.pyplot as plt
b=sns.countplot(data['G3']) #counts the number of students for a grade recieved
b.axes.set_title('Final grade of students',fontsize=30) #title of the plot
b.set_xlabel('Final Grade',fontsize=20) # x axis
b.set_ylabel('Count',fontsize=20)#y axis
plt.show
plt.figure(figsize=(18,16))
sns.heatmap(data.corr(),annot=True)
plt.show()

data.columns #returns all the columns(attributes) in the dataset
data.shape #returns the number of rows and columns in the dataset

##DATA CLEANING
data.isnull().any()#this block of code is used to check for any missing values in our dataset by column
data.isnull().sum() # this gives the total number of null values by column

#Gender count in the dataset
male_students=len(data[data['sex']=='M'])
female_students=len(data[data['sex']=='F'])
print('No. of male students',male_students)
print('No. of female students',female_students)

#cleaning data
#removing columns of data which are irrelevant to a students grade
# or does not affect the students grade in our model
# in this case School and age 
data.drop(['school','age','Pstatus'],axis=1,inplace=True)
data.columns

#Building the model/Encoding
#MAPPING to ensure we have binary data to work
# with our model coz we cannot use string data
# for machine learning
# yes/no values:
d = {'yes':1,'no':0}
data['schoolsup']=data['schoolsup'].map(d)
data['famsup']=data['famsup'].map(d)
data['paid']=data['paid'].map(d)
data['activities']=data['activities'].map(d)
data['nursery']=data['nursery'].map(d)
data['higher']=data['higher'].map(d)
data['internet']=data['internet'].map(d)
data['romantic']=data['romantic'].map(d)
# .map is a function for mapping
d={'F':1,'M':0}
data['sex']=data['sex'].map(d)
#mapping the parent's job
d={'teacher':0,'health':1,'services':2,'at_home':3,'other':4}
data['Mjob']=data['Mjob'].map(d)
data['Fjob']=data['Fjob'].map(d)
#mapping the reason data
d={'home':0,'reputation':1,'course':2,'other':3}
data['reason']=data['reason'].map(d)
#mapping the guardian data
d={'mother':0,'father':1,'other':2}
data['guardian']=data['guardian'].map(d)
#mapping the address - urban,rural
d={'R':0,'U':1}
data['address']=data['address'].map(d)
#mapping the familysize
d={'LE3':0,'GT3':1}
data['famsize']=data['famsize'].map(d)

data.dtypes

#Scaling the axes
# now we are using all the data in x axis as input 
# and G3 which is the output , we put that in y axis
# while training
from sklearn.model_selection import train_test_split
x=data.drop('G3',axis=1)
y=data['G3']
#we are using sklearn or scikit learn(library for machine learning)
# The sklearn library contains a lot of efficient tools for machine learning and statistical 
#modeling including classification, regression, clustering and dimensionality reduction.
data['G3']

data.isnull().any()
#before we train the data we need to be sure that there is no null values in our data
# after mapping so we check for null values in this code

#Loading Training and testing data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
#Building a model using svm(support vector machines)
#importing svm algorithm
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
import warnings
warnings.filterwarnings('ignore')

#Training the model
regressor.fit(X_train,y_train)
#Testing
y_pred=regressor.predict(X_test)
#RESULTS
print(regressor.score(X_test,y_test))
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)