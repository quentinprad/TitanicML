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



def DecisionTree(trainSet, trainLabels, testSet):
	
    # Train a single decision tree
    clf = DecisionTreeClassifier(max_depth=7)

    # Train the classifier
    clf.fit(trainSet, trainLabels)

    # Do classification on the test dataset and return 
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels

def RandomForestFeatures(trainSet, trainLabels, testSet):

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

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

def RandomForest(trainSet, trainLabels, testSet):
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=10,
                             random_state=0)
    
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)
    
    return predictedLabels



def logReg(trainSet, trainLabels, testSet):

    # Train 
    clf = LogisticRegression(C = 1e12, random_state= 33)

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
