import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from Pre_processing import *
from sklearn import metrics
import  time
from sklearn.feature_selection import SelectKBest, chi2


#import graphviz

data = pd.read_csv('VideoLikesDataset.csv')
#Drop the rows that contain missing values
ved_data=data.iloc[:,4:14]
X=data.iloc[:,4:13] #Features
Y=data['likes'] #Label
cols=('publish_time','tags','comments_disabled','ratings_disabled','video_error_or_removed','video_description')
X=Feature_Encoder(X,cols);
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)

model_filename = "Logistic_Regression.pickle"
my_knn_model = pickle.load(open(model_filename, 'rb'))
result = my_knn_model.predict(X_test)
print("mean error : ", metrics.mean_squared_error(y_test, result))



