import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from Pre_processing import *
import  time
from sklearn.feature_selection import SelectKBest, chi2


#import graphviz

data = pd.read_csv('VideoLikesTestingClassification.csv')
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

model_filename = "Model_svc.pickle"
my_knn_model = pickle.load(open(model_filename, 'rb'))
result = my_knn_model.predict(X_test)
accuracy=np.mean(result == Y_test)*100
print("accuracy :",accuracy)


