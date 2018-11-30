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
        return 1
    else:
        return 0
    

full['Child'] = full['Age'].apply(lambda size: child(size))
full['Child']

# Adding Mother variable

full['Mother']=0
def mother(i):
    if full['Sex'][i]=='female':
            if full['Parch'][i]>0:
                if full['Age'][i]>18:
                    if full['Title'][i]!= 'Miss':
                        full['Mother'][i] =1    #1 is a mum, 0 no
                    else:
                        full['Mother'][i]=0
                else:
                    full['Mother'][i]=0
            else:
                full['Mother'][i]=0
    else:
        full['Mother'][i]=0
                
                    

for i in range(len(full)):
    mother(i)
    

full['Mother'] #i made itttttttttttt

#create age times class feature
full['age_times_class']=full['Age']*full['Pclass']


#discretize sex
for i in range(len(full)):
    if full['Sex'][i]=='male':
        full['Sex'][i]=0
    else:
        full['Sex'][i]=1
    


#discretize fare
def fare(i):    
    for i in range(len(full)):
        if full['Fare'][i]<=7.91:
            full['Fare'][i]=0
        if full['Fare'][i]>7.91:
                if full['Fare'][i]<=14.454:
                    full['Fare'][i]=1
        if full['Fare'][i]>14.454:
            if full['Fare'][i]<31:
                full['Fare'][i]=2
        if full['Fare'][i]>31:
            full['Fare'][i]=3
                                
for i in range(len(full)):
    fare(i)

print(full['Fare']) 

def embarked(dataset):
    # embarked {S, C, Q} => 3 binary variables, dummies 
    embarked_separate_port = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
    dataset = pd.concat([dataset, embarked_separate_port], axis=1)
    return dataset.drop('Embarked', axis=1)
 
full = embarked(full) 
full.head()
#correlations
corr =full.corr()
print(corr)

import seaborn as sns
import matplotlib.pyplot as plt
 
sns.heatmap(np.abs(corr),xticklabels=corr.columns,yticklabels=corr.columns)


#decision tree for age or we're using the median?
target = full["Age"].values
features_one = full[["Pclass", "Sex", "Child", "Family_size","Fare","Parch","SibSp","Mother","age_times_class"]].values

#fitting the decision tree
nkd_tree = tree.DecisionTreeClassifier()
nkd_tree = nkd_tree.fit(features_one, target)
    
#observing the importance and score of the features
print(nkd_tree.feature_importances_)
print(nkd_tree.score(features_one, target))



#plots
bins = np.linspace(0, 80, 35)

# Two subplots, unpack the axes array immediately


