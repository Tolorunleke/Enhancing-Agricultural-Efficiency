import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn. metrics import accuracy_score, classification_report, precision_score, recall_score, log_loss, roc_curve, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, f_regression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder


#load relative dataset for training the regression model
abalone= pd.read_csv("/Users/user/Documents/Studies/Machine_Git/AbloneData/train.csv")
abalone_test= pd.read_csv("/Users/user/Documents/Studies/Machine_Git/AbloneData/test.csv")

#exploratory anaysis on Train and test data... starting with train data
data_length= len(abalone)
print("the total sample of train data is:", data_length)

#basic summary info
info= abalone.describe()
print(info) 

info2= abalone.info()

#check for outliers using the box plot and also visualizing the inter quartile ranges
box= sns.boxplot(abalone.iloc[:, 1:-1]) # there are outliers in the data

#check the given distribution of dataset with scatter plor
plt.title("The Abalone Data distribution and direction")
plt.xlabel("id")
sct= sns.scatterplot(data=abalone.iloc[:, 1: -1])
plt.show()

#check for null values 
nulls= abalone.isna().any()
print(nulls)


#preprecessing state- onehotencoder
abalone["Sex"].value_counts()
abalone["Sex"].describe()
abalone["Sex"].unique()

rings= abalone['Rings']
abalone= abalone.drop('Rings', axis=1)

#using one-hot encoder, a new object needs to be created
encoder= OneHotEncoder()
encoded= encoder.fit_transform(abalone[["Sex"]]).toarray()
encoder.get_feature_names_out()  #give the names of the new columns
sexdata= pd.DataFrame(encoded, columns= encoder.get_feature_names_out())

#repeat the same thing for test data
test_encoded= encoder.transform(abalone_test[["Sex"]]).toarray()
test_encoded= pd.DataFrame(test_encoded, columns= encoder.get_feature_names_out())

#new dataset for the training the model
abalone= abalone.drop("Sex", axis=1).join(sexdata)
abalone_test= abalone_test.drop('Sex', axis=1).join(test_encoded)


# preprocessing state- using robust scaler
scaler= RobustScaler()
train_robust_abalone= scaler.fit_transform(abalone)
test_robust_abalone= scaler.transform(abalone_test)

#preprocessing state- using standard sclaer()
scaler2= StandardScaler()
train_standard_abalone= scaler2.fit_transform(abalone)
trest_standard_abalone= scaler2.transform(abalone_test)


#using select k-best and estimators to select the best k for the model
selector= SelectKBest(f, k=7)
abalone_features= selector.fit_transform(train_robust_abalone, rings)
featurename= selector.get_support(indices= True)

#using recursive cross validation for selecting the best feature for the model
estimator= RandomForestRegressor(random_state= 40)
selector= RFECV(estimator, cv= 4)
abalone_features_cv= selector.fit_transform(train_robust_abalone, rings)
test_robust_abalone= selector.transform(test_robust_abalone)


#see the names of the features selected
featurename= selector.get_support(indices= True)
names= abalone.columns[featurename]

#split data into train and test sets
Xtrain, Xtest, ytrain, ytest= train_test_split(abalone_features_cv, rings, test_size=0.2, random_state=40)

#Train model and check the rmse as requested in competition brief
model= AdaBoostRegressor()
train_model= model.fit(Xtrain, ytrain)
predictions= model.predict(Xtest)

#evaluate the predictions
from sklearn import metrics
score= model.score(Xtest, ytest)
mean_error= mean_squared_error(predictions, ytest)


#using the ranmdom forest regressor
rf_model= RandomForestRegressor(random_state=10)
rf_train= rf_model.fit(Xtrain, ytrain)
y_predicted= rf_model.predict(Xtest)
#evaluate random forest
mean_error= mean_squared_error(y_predicted, ytest)
score= rf_model.score(Xtest, ytest)
root_mean= mean_squared_log_error(y_predicted, ytest)**0.5
print(f"Random Forest Regressor results: \n Score: {score} \n Mean Error: {mean_error} \n Mean error: {root_mean} ")


#using gradient boosting regressor
gb_model= GradientBoostingRegressor(random_state= 10)
gb_train= gb_model.fit(Xtrain, ytrain)
y_predicted= gb_model.predict(Xtest)
#evaluate the performance
score= gb_model.score(Xtest, ytest)
mean_error= mean_squared_error(y_predicted, ytest)
root_mean_error= mean_squared_log_error(y_predicted, ytest) ** 0.5
print(f"Gradient boosting regressor results: \n Score: {score} \n Mean Error: {mean_error} \n Mean error: {root_mean_error} ")


#using knn
k_model= KNeighborsRegressor()
k_train= k_model.fit(Xtrain, ytrain)
y_predicted= k_model.predict(Xtest)
#evaluate performance
score= k_model.score(Xtest, ytest)
mean_error= mean_squared_error(y_predicted, ytest)
root_mean_error= mean_squared_log_error(y_predicted, ytest) ** 0.5
print(f"KNeighbors Regressor results: \n Score: {score} \n Mean Error: {mean_error} \n Mean error: {root_mean_error} ")
