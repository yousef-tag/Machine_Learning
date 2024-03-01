import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from Pre_processing import *
from sklearn.feature_selection import SelectKBest, chi2
import  time


#import graphviz
data = pd.read_csv('VideoLikesDatasetClassification.csv')
X=data.iloc[:,2:13]
target = data['VideoPopularity']
Y=[]
for val in target:
    if(val == 'High'):
        Y.append(2)
    elif(val == 'Medium'):
        Y.append(1)
    else:
        Y.append(0)
print(Y)
cols=('title','channel_title','publish_time','tags','comments_disabled','ratings_disabled','video_error_or_removed','video_description')
X=Feature_Encoder(X,cols);
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)
##################################################################################################################
AdaBoostClassifierTrainningStartTime = time.time()  ######### AdaBoost Classifier Training Start Time
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME",
                         n_estimators=100)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
bdt.fit(X_train,Y_train)
AdaBoostClassifierTrainningENDTIME = time.time()   ##### AdaBoost Classifier Training END TIME

AdaBoostClassifierTestingStartTIME = time.time()  #######  AdaBoost Classifier Testing Start TIME
X_test = scaler.transform(X_test)
Y_prediction = bdt.predict(X_test)
AdaBoostClassifierTestingENDTIME = time.time()  #######  AdaBoost Classifier Testing END TIME


accuracy=np.mean(Y_prediction == Y_test)*100
Adaboost_accuracy=accuracy   ########  Adaboost accuracy
print ("The achieved accuracy using Adaboost is " + str(accuracy))


model_filename = "bdt.pkl"
saved_model = pickle.dump(bdt, open(model_filename,'wb'))
print('Model bdt is saved into to disk successfully Using Pickle')

############################################################################################

Decision_Tree_StartTrainningTime=time.time()  ############## Decision_Tree Start Training Time
clf = tree.DecisionTreeClassifier(max_depth=20)
clf.fit(X_train,Y_train)
Decision_Tree_EndTrainningTime=time.time()   ####### Decision Tree End Training Time

Decision_Tree_StartTestingTime=time.time()  ######### Decision Tree Start Testing Time
Y_prediction = clf.predict(X_test)
Decision_Tree_EndTestingTime=time.time()   ########## Decision Tree End Testing Time


accuracy=np.mean(Y_prediction == Y_test)*100
Decision_Tree_accuracy=accuracy   ########Decision Tree accuracy
print ("The achieved accuracy using Decision Tree is " + str(accuracy))


model_filename = "Tree.pickle"
saved_model = pickle.dump(clf, open(model_filename,'wb'))
print('Model decision tree is saved into to disk successfully Using Pickle')
###########################################################################################################



#######################################################################################
#plotting of accuracy
modelNames=['Adaboost','Decision Tree']
modelAccuracy=[Adaboost_accuracy,Decision_Tree_accuracy]
plt.bar(modelNames,modelAccuracy)
plt.title('Accuracy Bar')
plt.ylabel('Accuracy Percentage')
plt.xlabel('Epochs Number')
plt.show()


###################################################################################################

Decision_Tree_TrainningTime = Decision_Tree_EndTrainningTime-Decision_Tree_StartTrainningTime

AdaBoost_TrainningTime = AdaBoostClassifierTrainningENDTIME - AdaBoostClassifierTrainningStartTime

#plotting of training time
modelNames=['Adaboost','Decision Tree']
modelAccuracy=[AdaBoost_TrainningTime,Decision_Tree_TrainningTime]
plt.bar(modelNames,modelAccuracy)
plt.title('Training Time Bar')
plt.ylabel('Training time in second(s)')
plt.xlabel('Epochs Number')
plt.show()


#################################################################################################
Decision_Tree_TestingTime=Decision_Tree_EndTestingTime-Decision_Tree_StartTestingTime

Adaboost_TestingTime=AdaBoostClassifierTestingENDTIME - AdaBoostClassifierTestingStartTIME

#plotting of testing time
modelNames=['Adaboost','Decision Tree']
modelTestTime=[Adaboost_TestingTime,Decision_Tree_TestingTime]
plt.bar(modelNames,modelAccuracy)
plt.title('Testing Time Bar')
plt.ylabel('Testing Time in second(s)')
plt.xlabel('Epochs Number')
plt.show()


