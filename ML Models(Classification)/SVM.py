import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from Pre_processing import *
from sklearn.feature_selection import SelectKBest, chi2

data = pd.read_csv('VideoLikesDatasetClassification.csv')
data = data.fillna(data.mean())
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0,shuffle=True)
print(X_train)

# we create an instance of SVM and fit out data.
C = 10  # SVM regularization parameter

Normal_SVS_STtrain=time.time() ###normal svc train Start time

svc=svm.SVC(random_state=1,C=C).fit(X_train,Y_train)

Normal_SVS_ETtrain=time.time()     # ### normal svc train end time

Normal_SVS_STtest=time.time()       # ## normal svc test Start time

svc_acc=svc.score(X_test,Y_test)

Normal_SVS_ETtest=time.time()         # ##normal svc test end time

print("Normal SVC accuracy is :",svc_acc)
Normal_SVS_Accuracy=svc_acc *100        #   ##### normal svc accuracy

#####################################################################################################
Linear_SVS_STtrain=time.time()          # ##Linear svc train Start time

lin_svc=svm.LinearSVC(C=C).fit(X_train,Y_train)

Linear_SVS_ETtrain=time.time()          # ##Linear svc train end time
Linear_SVS_STtest=time.time()          # ##Linear svc test Start time

lin_svc_acc=lin_svc.score(X_test,Y_test)
Linear_SVS_ETtest=time.time()          # ##Linear svc test end time

print("linear SVC accuracy is :",lin_svc_acc)
Linear_SVS_Accuracy=lin_svc_acc * 100         #   ##### linear svc accuracy


##########################################################################################################
RBF_SVS_STtrain=time.time()          # ##RBF svc train Start time

rbf_svc=svm.SVC(kernel='rbf',gamma=0.8,C=C).fit(X_train,Y_train)

RBF_SVS_ETtrain=time.time()          # ##RBF svc train end time
RBF_SVS_STtest=time.time()          # ##RBF svc test Start time
rbf_acc=rbf_svc.score(X_test,Y_test)
RBF_SVS_ETtest=time.time()          # ## RBF svc test end time

print("RBF SVC accuracy is :",rbf_acc)
RBF_SVS_Accuracy = rbf_acc * 100         #   ##### RBF svc accuracy

###########################################################################################################
Poly_SVS_STtrain=time.time()          # ## Polynomial svc train Start time

poly_svc=svm.SVC(kernel='poly',degree=1,C=C).fit(X_train,Y_train)

Poly_SVS_ETtrain=time.time()          # ## Polynomial svc train end time
Poly_SVS_STtest=time.time()          # ## Polynomial svc test Start time

poly_acc=poly_svc.score(X_test,Y_test)

Poly_SVS_ETtest=time.time()          # ## Polynomial svc test end time

print("Polynomail SVC degree 3 accuracy is :",poly_acc)
Poly_SVS_Accuracy = poly_acc * 100        #   ##### Polynomial svc accuracy

###############################################################################################################
#save 4 model

model_filename = "Model_svc.pickle"
saved_model = pickle.dump(svc, open(model_filename,'wb'))
print('Model Normal svm is saved into to disk successfully Using Pickle')


model_filename = "Model_linear_SVM.pickle"
saved_model = pickle.dump(lin_svc, open(model_filename,'wb'))
print('Model linear svm is saved into to disk successfully Using Pickle')

model_filename = "Model_rbf_SVM.pickle"
saved_model = pickle.dump(rbf_svc, open(model_filename,'wb'))
print('Model rbf is saved into to disk successfully Using Pickle')


model_filename = "Model_polynomial_SVM.pickle"
saved_model = pickle.dump(poly_svc, open(model_filename,'wb'))
print('Model polonmial svm is saved into to disk successfully Using Pickle')


######################################################################################
#plotting of accuracy
modelNames=['Normal SVC','Linear SVC','RBF SVC','Polynomial SVC']
modelAccuracy=[Normal_SVS_Accuracy,Linear_SVS_Accuracy,RBF_SVS_Accuracy,Poly_SVS_Accuracy]
plt.bar(modelNames,modelAccuracy)
plt.title('Accuracy Bar')
plt.ylabel('Accuracy Percentage')
plt.xlabel('Epochs Number')
plt.show()

########################################################################

Normal_SVC_Training = Normal_SVS_ETtrain - Normal_SVS_STtrain
Linear_SVC_Training = Linear_SVS_ETtrain - Linear_SVS_STtrain
RBF_SVC_Training = RBF_SVS_ETtrain - RBF_SVS_STtrain
Poly_SVC_Training = Poly_SVS_ETtrain -Poly_SVS_STtrain


#plotting of training time
modelNames=['Normal SVC','Linear SVC','RBF SVC','Polynomial SVC']
modelAccuracy=[Normal_SVC_Training,Linear_SVC_Training,RBF_SVC_Training,Poly_SVC_Training]
plt.bar(modelNames,modelAccuracy)
plt.title('Training Time Bar')
plt.ylabel('Training time in second(s)')
plt.xlabel('Epochs Number')
plt.show()


###################################################################

Normal_SVC_Testing = Normal_SVS_ETtest - Normal_SVS_STtest
Linear_SVC_Testing = Linear_SVS_ETtest - Linear_SVS_STtest
RBF_SVC_Testing = RBF_SVS_ETtest - RBF_SVS_STtest
Poly_SVC_Testing = Poly_SVS_ETtest - Poly_SVS_STtest

#plotting of testing time
modelNames=['Normal SVC','Linear SVC','RBF SVC','Polynomial SVC']
modelTestTime=[Normal_SVC_Testing ,Linear_SVC_Testing , RBF_SVC_Testing ,Poly_SVC_Testing]
plt.bar(modelNames,modelAccuracy)
plt.title('Testing Time Bar')
plt.ylabel('Testing Time in second(s)')
plt.xlabel('Epochs Number')
plt.show()




