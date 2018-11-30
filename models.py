from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import l1_min_c
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC 
from sklearn.model_selection import *




def Scaledata(Set):
    
    scaler = StandardScaler()
    scaler.fit(Set)
    
    return Set


def DecisionTree(trainSet, trainLabels, testSet):
	
    # Train a single decision tree
    clf = DecisionTreeClassifier(max_depth=7)

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
    sfm = SelectFromModel(clf, threshold=0.04)
    # Train the selector
    sfm.fit(trainSet, trainLabels)
    trainSet_trans = sfm.transform(trainSet)
    testSet_trans = sfm.transform(testSet)
    clf.fit(trainSet_trans, trainLabels)
    predictedLabels = clf.predict(testSet_trans)
    
    return predictedLabels


def LogReg(trainSet, trainLabels, testSet):

    # Train 
    clf = LogisticRegression(penalty='l1',n_jobs =-1,solver='liblinear',C=1)
    clf.fit(trainSet,trainLabels)

    predictedLabels = clf.predict(testSet)

    return predictedLabels


    

def lda(trainSet, trainLabels, testSet):
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

def knn(trainSet, trainLabels, testSet):
    
    clf = KNeighborsClassifier()
    
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

    
def RandomForest(trainSet, trainLabels, testSet):
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)
    clf.fit(trainSet, trainLabels)
    print("FEATURE IMPORTANCE:")
    print(clf.feature_importances_) 
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels


    
   
def RandomForestFeatures(trainSet, trainLabels, testSet):



    # Build a forest and compute the feature importances

    forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



    forest.fit(trainSet, trainLabels)

    importances = forest.feature_importances_

    stddev = std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

    indices = argsort(importances)[::-1]



    # Print the feature ranking

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

    clf = RandomForestClassifier(class_weight=None, max_depth=10, max_features='auto', n_estimators=400)
    # Create a selector object that will use the random forest classifier to identify
    # features that have an importance of more than 0.15
    sfm = SelectFromModel(clf, threshold=0.05)
    # Train the selector
    sfm.fit(trainSet, trainLabels)
    trainSet_trans = sfm.transform(trainSet)
    testSet_trans = sfm.transform(testSet)
    clf.fit(trainSet_trans, trainLabels)
    predictedLabels = clf.predict(testSet_trans)
    
    return predictedLabels

    
    
    
    
def GaussNaiveBayes(trainSet, trainLabels, testSet):
    
    gnb = GaussianNB()
    gnb.fit(trainSet, trainLabels)
    predictedLabels = gnb.predict(testSet)
    return predictedLabels

    

def AverageModels(trainSet, trainLabels, testSet):
    
    randfor = RandomForest(trainSet, trainLabels, testSet)
    gauss = gaussNaiveBayes(trainSet, trainLabels, testSet)
    loreg = logReg(trainSet, trainLabels, testSet)
    ldaa = lda(trainSet, trainLabels, testSet)
    
    average_variable = (randfor+loreg+ldaa+randfor) / 4.0
    for i in range(len(average_variable)):
        if average_variable[i] < 0.5:
            average_variable[i] = 0
        else:
            average_variable[i] = 1
    return average_variable

        
    
#def test_rf(trainSet, trainLabels, testSet):
#    
#    clf = Pipeline([
#            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#            ('classification', RandomForestClassifier())
#                                                        ])
#    clf.fit(trainSet, trainLabels)
#    predictedLabels = clf.predict(testSet)
#    
#    return predictedLabels
#    
    


def Tuning(trainSet, trainLabels):
   import numpy as np
# Number of trees in random forest
   n_estimators = [int(x) for x in np.linspace(start = 350, stop = 450, num = 10)]
# Number of features to consider at every split
   max_features = ['auto']
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
   rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
   rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
   rf_random.fit(trainSet, trainLabels)

   print('best params = ', rf_random.best_params_)

   return rf_random.best_params_ 
    
    
