from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

def DecisionTree(trainSet, trainLabels, testSet):
	
    # Train a single decision tree
    clf = DecisionTreeClassifier(max_depth=7)

    # Train the classifier
    clf.fit(trainSet, trainLabels)

    # Do classification on the test dataset and return 
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
