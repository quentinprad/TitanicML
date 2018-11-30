from numpy import *
from sklearn import cross_validation
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import *

test = pd.read_csv('/Users/quentinpradelle/Desktop/EssecCentrale/Cours Centrale/Machine Learning/AssignmentsAndProject/ML-DSBA-AI-Assignment_2/Data/test.csv')
train = pd.read_csv('/Users/quentinpradelle/Desktop/EssecCentrale/Cours Centrale/Machine Learning/AssignmentsAndProject/ML-DSBA-AI-Assignment_2/Data/train.csv')
full = train.append(test, ignore_index = True)
X_labels = train['Survived']
X_labels = X_labels.values

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

# Get only first letter for Cabin Code ands replace Null values by 'U'
full['CabinCode'] = full[full['Cabin'].notnull()].Cabin.astype(str).str[0]
full['CabinCode'].replace(NaN, 'U', inplace=True)

# Separate Title and FamilyName from Name column
full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full['FamilyName'] = full.Name.str.extract(' ([A-Za-z]+),', expand = False)
full['Title'].value_counts()

# Transform different titles with int
full['Title'] = full['Title'].map({'Mr' : 0, 'Miss' : 1, 'Mrs' : 2, 'Master' : 3, 'Col' : 4, 'Don' : 4, 'Dr' : 4, 'Jonkheer' : 4, 'Major' : 4, 'Sir' : 4, 'Dona' : 4, 'Lady' : 4, 'Mlle' : 4, 'Ms' : 4, 'Countess' : 4, 'Mme' : 4, 'Capt' : 4, 'Rev' : 4}).astype(int)

# Check visually if Titles seem to be linked with survival
SurvivalByTitle = full['Title'].corr(full['Survived'])
SurvivalbyTitle = full.groupby('Title')

for key, item in SurvivalbyTitle:
    print(SurvivalbyTitle.get_group(key), "\n\n")
    
# Create the column child, and indicate whether child or adult
full['Child'] = (full['Age'] < 18) * 1

# Creating the column Mother 
full['Mother'] = ((full.Sex == 1) & (full.Parch > 0) & (full.Title != 1)) * 1

# Create a column Family size and remove Parch and SibSp
full['FamilySize'] = full['Parch'] + full['SibSp'] + 1
full.drop(['Parch', 'SibSp'], axis=1, inplace=True)

# Transform missing ages with the mean of similar passengers
full['Age'] = full.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'].fillna((full['Age'].median()), inplace=True)

# Create age * class feature
full['age_times_class']=full['Age']*full['Pclass']

# Transform Cabin Codes with numbers
full['CabinCode'].value_counts()
full['CabinCode'] = full['CabinCode'].map({'U' : 0, 'C' : 1, 'B' : 2, 'D' : 3, 'E' : 4, 'A' : 5, 'F' : 6, 'G' : 7, 'T' : 8}).astype(int)

# Transform Embarked column with numbers
full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2}).astype(int)
full = pd.get_dummies(full, prefix = 'dummy_Embarked', columns = ['Embarked'], drop_first = True)

# Transform Title column to dummy
full = pd.get_dummies(full, prefix = 'dummy_Title', columns = ['Title'], drop_first = True)

# Remove the unneccessary column 
full.drop(['Cabin', 'FamilyName', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

full['AgeBin'] = pd.cut(full['Age'].astype(int), 4, labels=[0, 1, 2, 3])
full['AgeBin'] = full['AgeBin'].astype(int)

# Check if there are nan values remaining / 418 are normal (test set survived column), 1 is not normal 
full.isnull().values.any()
full.isnull().sum().sum()

# Save the cleaned dataframe as a csv file
full.to_csv("dfTitanicCleanFull.csv", sep = ',')

# Separate test and train set from the cleaned full set
full = full.convert_objects(convert_numeric = True)
full = full.values 

# Save the cleaned numpy array as a csv file
savetxt("titanicCleanFull.csv", full, delimiter=",")


# Convert nan value in age to median value
#full[979, 0] = median(full, axis = 1)[0]

X_train = full[:891, :]
X_test = full[891:, :]
Y_train = X_train[:, 4] # Survived column

X_train = delete(X_train, 4, 1)
X_test = delete(X_test, 4, 1)

# After analysis, we discovered that the four most important features were

#X_train = delete(X_train, 8, 1)
#X_test = delete(X_test, 8, 1)

#X_train = delete(X_train, 1, 1)
#X_test = delete(X_test, 1, 1)
#
#X_train = delete(X_train, 5, 1)
#X_test = delete(X_test, 5, 1)

#________________________________________________________________________________________________________________

# Initialize cross validation
kf = cross_validation.KFold(X_train.shape[0], n_folds=20)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  

# Loop for testing the
for trainIndex, testIndex in kf:
    trainSet = X_train[trainIndex]
    testSet = X_train[testIndex]
    trainLabels = Y_train[trainIndex]
    testLabels = Y_train[testIndex]
	
    predictedLabels = RandomForest(trainSet, trainLabels, testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
    
    print 'Accuracy: ' + str(float(correct)/(testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print 'Total Accuracy: ' + str(totalCorrect/float(totalInstances))

# Now, let's do the prediction for the Kaggle test set

Y_predict = zeros((418, 2))
for i in range(0, 418):
    Y_predict[i, 0] = 892 + i
    Y_train = Y_train.astype(float)
Y_predict[:, 1] = RandomForest(X_train, Y_train, X_test)

Y_predict = Y_predict.astype(int)

final_result = pd.DataFrame({'PassengerId': Y_predict[:, 0], 'Survived': Y_predict[:, 1]})

final_result.to_csv('submission.csv', index = False)