import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
import pickle
from Pre_processing import *
from sklearn.feature_selection import SelectKBest, chi2
import time
from sklearn.model_selection import train_test_split
#Read dataset`
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
X = SelectKBest(chi2, k=4).fit_transform(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)
######################################################################

TrainingStartTime_knn=time.time()
knn = KNN(n_neighbors = 10)
knn.fit(X_train, Y_train)
TrainingEndTime_knn=time.time()
TestingStartTime_knn=time.time()
Y_prediction = knn.predict(X_test)
accuracy=np.mean(Y_prediction == Y_test)*100
knn_accuracy=accuracy
print("KNN accuracy :",knn_accuracy)
TestingEndtTime_knn=time.time()

#################################################################
# Fit the model on training set

TrainingStartTime_logistic=time.time()  ####### Training Start Time
model = LogisticRegression()
model.fit(X_train, Y_train)
TrainingEndTime_logistic=time.time()  ####### Training End Time

TestingStartTime_logistic=time.time()  ####### testing Start Time

accuracy1=model.score(X_test,Y_test)

Accuracy_of_logistic=accuracy1 ######Accuracy

print("logistic Regression accuracy is ",Accuracy_of_logistic)
Y_predicted=model.predict(X_test)
cm=confusion_matrix(Y_test,Y_predicted)

TestingEndtTime_logistic=time.time()  ####### testing End Time

print("confusion Matrix :")
print(cm)

#############################################################################
# save the model to disk

model_filename = "Knn.pickle"
saved_model = pickle.dump(knn, open(model_filename,'wb'))
print('Model KNN is saved into to disk successfully Using Pickle')


model_filename = "Logistic_Regression.pickle"
saved_model = pickle.dump(model, open(model_filename,'wb'))
print('Model3 is saved into to disk successfully Using Pickle')

##############################################################################################
#plotting of accuracy
modelNames=['KNN']
modelAccuracy=[knn_accuracy]
plt.bar(modelNames,modelAccuracy)
plt.title('Accuracy Bar')
plt.ylabel('Accuracy Percentage')
plt.xlabel('Epochs Number')
plt.show()

modelNames=['Logistic Regression']
modelAccuracy=[Accuracy_of_logistic]
plt.bar(modelNames,modelAccuracy)
plt.title('Accuracy Bar')
plt.ylabel('Accuracy Percentage')
plt.xlabel('Epochs Number')
plt.show()


TrainingTime_logistic = TrainingEndTime_logistic - TrainingStartTime_logistic
TrainingTime_knn=TrainingEndTime_knn-TrainingStartTime_knn
#plotting of training time
modelNames=['Logistic Regression','KNN']
modelAccuracy=[TrainingTime_logistic,TrainingTime_knn]
plt.bar(modelNames,modelAccuracy)
plt.title('Training Time Bar')
plt.ylabel('Training time in second(s)')
plt.xlabel('Epochs Number')
plt.show()

###################################################################################################
TestingTime_logistic = TestingEndtTime_logistic -TestingStartTime_logistic
TestingTime_knn = TestingEndtTime_knn -TestingStartTime_knn
#plotting of testing time
modelNames=['Logistic Regression','KNN']
modelTestTime=[TestingTime_logistic,TestingTime_knn]
plt.bar(modelNames,modelAccuracy)
plt.title('Testing Time Bar')
plt.ylabel('Testing Time in second(s)')
plt.xlabel('Epochs Number')
plt.show()


