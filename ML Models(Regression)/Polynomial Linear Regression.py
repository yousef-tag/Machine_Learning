import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
import pickle
#Load players data
data = pd.read_csv('VideoLikesDataset.csv')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
data.dropna(how='any',inplace=True)
ved_data=data.iloc[:,4:14]
X=data.iloc[:,4:13] #Features
Y=data['likes'] #Label
cols=('publish_time','tags','comments_disabled','ratings_disabled','video_error_or_removed','video_description')
X=Feature_Encoder(X,cols);

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=2002,test_size = 0.30)
#Get the correlation between the features
corr = ved_data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['likes']>0.5)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = ved_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))


print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

model_filename = "polynomial.pickle"
saved_model = pickle.dump(poly_model, open(model_filename,'wb'))
print('Model regression is saved into to disk successfully Using Pickle')