from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv('/Users/quentinpradelle/Desktop/EssecCentrale/Cours Centrale/Machine Learning/AssignmentsAndProject/ML-DSBA-AI-Assignment_2/Data/test.csv')
train = pd.read_csv('/Users/quentinpradelle/Desktop/EssecCentrale/Cours Centrale/Machine Learning/AssignmentsAndProject/ML-DSBA-AI-Assignment_2/Data/train.csv')
full = train.append(test, ignore_index = True)

print('Dataset \'full\' shape:', full.shape)

# Perform initial data exploration 
# Count the nulls

names = full.columns
null_count = full.isnull().sum()
null_pct = full.isnull().mean()
null_count.name = 'Null Count'
null_pct.name = 'Null Percent'
nulls = pd.concat([null_count, null_pct], axis=1)
nulls

# Check passengers with null values for fare or embarked 
full[full['Embarked'].isnull() | full['Fare'].isnull()]

# Fill the missing value for fare with median value and for embarked with the modal values
full['Fare'] = full.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.median()))

full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])
full['Embarked'].value_counts()

# Transform sex column to 0s and 1s
full['Sex'] = full['Sex'].map({'male' : 0, 'female' : 1}).astype(int)

# Create a column Family size and remove Parch and SibSp
full['FamilySize'] = full['Parch'] + full['SibSp'] + 1
full.drop(['Parch', 'SibSp'], axis=1, inplace=True)

# Get only first letter for Cabin Code ands replace Null values by 'U'
full['CabinCode'] = full[full['Cabin'].notnull()].Cabin.astype(str).str[0]
full['CabinCode'].replace(NaN, 'U', inplace=True)

# Separate Title and FamilyName from Name column
full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full['FamilyName'] = full.Name.str.extract(' ([A-Za-z]+),', expand = False)
full['Title'].value_counts()

# Transform different titles with int
full['Title'] = full['Title'].map({'Mr' : 0, 'Miss' : 1, 'Mrs' : 2, 'Capt' : 3, 'Col' : 4, 'Don' : 5, 'Dr' : 6, 'Jonkheer' : 7, 'Major' : 8, 'Sir' : 9, 'Dona' : 10, 'Lady' : 11, 'Mlle' : 12, 'Ms' : 13, 'Countess' : 14, 'Mme' : 15, 'Master' : 16, 'Rev' : 17}).astype(int)

# Check visually if Titles seem to be linked with survival
SurvivalByTitle = full['Title'].corr(full['Survived'])
SurvivalbyTitle = full.groupby('Title')

for key, item in SurvivalbyTitle:
    print(SurvivalbyTitle.get_group(key), "\n\n")

# Transform missing ages with the mean of similar passengers
full['Age'] = full.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

# Transform Cabin Codes with numbers
full['CabinCode'].value_counts()
full['CabinCode'] = full['CabinCode'].map({'U' : 0, 'C' : 1, 'B' : 2, 'D' : 3, 'E' : 4, 'A' : 5, 'F' : 6, 'G' : 7, 'T' : 8}).astype(int)

# Transform Embarked column with numbers
full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2}).astype(int)

# Remove the unneccessary column 
full.drop(['Cabin', 'FamilyName', 'Name', 'Ticket'], axis=1, inplace=True)

