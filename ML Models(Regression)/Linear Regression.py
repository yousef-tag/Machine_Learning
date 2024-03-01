import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Pre_processing import *
import pickle

#Load players data
data = pd.read_csv('VideoLikesDataset.csv')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
ved_data=data.iloc[:,4:14]
X=data.iloc[:,4:13] #Features
Y=data['likes'] #Label
cols=('publish_time','tags','comments_disabled','ratings_disabled','video_error_or_removed','video_description')
X=Feature_Encoder(X,cols);

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)
#Get the correlation between the features
corr = ved_data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['likes']>0.5)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = ved_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

cls = linear_model.LinearRegression()
cls.fit(X_train,y_train)
prediction= cls.predict(X_test)



print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

model_filename = "Logistic_Regression.pickle"
saved_model = pickle.dump(cls, open(model_filename,'wb'))
print('Model regression is saved into to disk successfully Using Pickle')