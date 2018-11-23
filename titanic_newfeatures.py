# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:44:48 2018

@author: Rebecca
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("C:\\Users\\Rebecca\\Desktop\\machinelearning\\titanic\\train.csv", delimiter=',' )
test = pd.read_csv("C:\\Users\\Rebecca\\Desktop\\machinelearning\\titanic\\test.csv", delimiter=',' )
print(train)
#train=pd.DataFrame(train)
#test=pd.DataFrame(test)
print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)
len(train) #891
len(test) #418
print('Test/Validation columns with null values:\n', test.isnull().sum())
print("-"*10)
full=pd.concat([train, test], ignore_index=True)
print(full)
len(full)
train.describe()
total = train.isnull().sum().sort_values(ascending=False)
percent_1 = train.isnull().sum()/train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
full.describe()


#create title column and fix rare titles
full['Title'] = full['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print(full['Title'].value_counts()) #keep Mr,Miss,Mrs,Master
full.replace({'Title':{'Mlle': 'Miss', 'Ms': 'Miss','Mme':'Mrs'}})
full.replace({'Title':{'Dona': 'rare_title', 'Lady': 'rare_title','the Countess':'rare_title','Capt':'rare_title','Col':'rare_title','Don':'rare_title','Dr':'rare_title','Major':'rare_title','Rev':'rare_title','Sir':'rare_title','Jonkheer':'rare_title'}})  

#exctract surname from name
full['Surname'] = full['Name'].apply(lambda x: str(x).split('.')[0].split(' ')[0])
full['Surname'] = full.Surname.str.replace(",", "")
full['Surname']


full=pd.DataFrame(full)
#Family size
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1

# Discretize family size
def conv_discrete (size):
    if (size == 1):
        return 'singleton'
    elif (1 < size <= 4) :
        return 'small'
    elif (size > 4):
        return 'large'
    else:
        return 'unspecified'
    
# Discretize family size
full['FamilyD'] = full['FamilySize'].apply(lambda size: conv_discrete(size))


# Create a Deck variable. Get passenger deck A - F:
full["Deck"] =full["Cabin"].str.slice(0,1)
full['Deck']

# Create the column child, and indicate whether child or adult
def child(size):
    if (size<18):
        return 'Child'
    else:
        return 'Adult'
    

full['Child'] = full['Age'].apply(lambda size: child(size))
full['Child']
#create a mother feature-> doesn't work yet
new = full[['Sex', 'Parch', 'Age','Title']].copy()
x=['sex','parch','age','title']
# Adding Mother variable
def mother(full,size):
    if (full['Sex']=='female'&fill['Parch']>0&full['Age']>18&full['Title']!='Miss'):
        size=1
    else:
        size=0

size_=full['Mother]
mother(full,size_)


full['Mother'] = new.apply(lambda x: mother(x))
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

#create age times class feature
#fill missing values 


