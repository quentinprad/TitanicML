from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC  
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def ScaleData(Set):

   # This function will normalize the data through z = (data - mean) / std
   scaler = StandardScaler()
   scaler.fit(Set)

   return Set


def DecisionTree(trainSet, trainLabels, testSet):
	
    # Train a single decision tree
    clf = DecisionTreeClassifier()
    # Train the classifier
    clf.fit(trainSet, trainLabels)
    # Do classification on the test dataset and return 
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

def DecisionTreeWithFeatureSelection(trainSet, trainLabels, testSet):

  # Train a single decision tree
   clf = DecisionTreeClassifier(max_depth=7)

  # Create a selector object that will use the random forest classifier to identify
   # features that have an importance of more than 0.05
   sfm = SelectFromModel(clf, threshold=0.05)
   
   # Train the selector
   sfm.fit(trainSet, trainLabels)
   trainSet_trans = sfm.transform(trainSet)
   testSet_trans = sfm.transform(testSet)
   clf.fit(trainSet_trans, trainLabels)
   predictedLabels = clf.predict(testSet_trans)

   return predictedLabels

def RandomForest(trainSet, trainLabels, testSet):
    
    clf = RandomForestClassifier(n_estimators = 50, max_features = 'log2', bootstrap = False, max_depth = 10, min_samples_leaf = 4, random_state = 0)
    clf.fit(trainSet, trainLabels)

    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

def RandomForestFeatures(trainSet, trainLabels, testSet):

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators = 250, random_state=0)

    # Fit the model and sort the features by importance
    forest.fit(trainSet, trainLabels)
    importances = forest.feature_importances_
    stddev = std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = argsort(importances)[::-1]

    # Print the features ranking
    print("Feature ranking:")

    for f in range(trainSet.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(trainSet.shape[1]), importances[indices], color="r", yerr=stddev[indices], align="center")
    plt.xticks(range(trainSet.shape[1]), indices)
    plt.xlim([-1, trainSet.shape[1]])
    plt.show()


def RandomForestWithFeatureSelection(trainSet, trainLabels, testSet):

   clf = RandomForestClassifier(n_estimators = 450, max_features = 'log2', bootstrap = False, max_depth = 75, min_samples_leaf = 4)
   # Create a selector object that will use the random forest classifier to identify
   
   # features that have an importance of more than 0.15
   sfm = SelectFromModel(clf, threshold=0.02)
   
   # Train the selector
   sfm.fit(trainSet, trainLabels)
   
   trainSet_trans = sfm.transform(trainSet)
   testSet_trans = sfm.transform(testSet)
   
   clf.fit(trainSet_trans, trainLabels)
   
   predictedLabels = clf.predict(testSet_trans)

   return predictedLabels


def LogReg(trainSet, trainLabels, testSet):

    # Train 
    clf = LogisticRegression(C = 1e12, penalty = 'l2', random_state= 33)
    clf.fit(trainSet,trainLabels)
    predictedLabels = clf.predict(testSet)

    return predictedLabels


def Knn(trainSet, trainLabels, testSet):
    
    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(trainSet, trainLabels)
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

def SVM(trainSet, trainLabels, testSet):
    
    clf = SVC() 
    clf.fit(trainSet, trainLabels)
    supportVectors = clf.support_vectors_
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

def Boosting(trainSet, trainLabels, testSet):
    
    clf = GradientBoostingClassifier(n_estimators=119, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(trainSet, trainLabels)
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels


def tuning(trainSet, trainLabels, testSet):
    import numpy as np
# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 150, stop = 450, num = 10)]
# Number of features to consider at every split
    max_features = ['auto', 'log2']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 21)]
    max_depth.append(None)
# Minimum number of samples required to split a node
    # 'min_samples_split': min_samples_split,min_samples_split = 0.5
    
# Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
    bootstrap = [True, False]
# Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
    rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
    rf_random.fit(trainSet, trainLabels)
    
    print('best params = ', rf_random.best_params_)

    return rf_random.best_params_

    
def averageModels(trainSet, trainLabels, testSet):
    
    logreg = LogReg(trainSet, trainLabels, testSet)
    decisiontree = DecisionTree(trainSet, trainLabels, testSet)
    randomforest = RandomForest(trainSet, trainLabels, testSet)
    svma = SVM(trainSet, trainLabels, testSet)
    Boost = Boosting(trainSet, trainLabels, testSet)
        
    averageModels = (Boost + logreg + randomforest + decisiontree + randomforest) / 5.0
    for i in range(len(averageModels)):
        if averageModels[i] < 0.5:
            averageModels[i] = 0
        else:
            averageModels[i] = 1
    
    return averageModels
    