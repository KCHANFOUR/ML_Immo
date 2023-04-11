import pandas as pd
import plotly.express as px
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# Paris Housing Prices

full_df = pd.read_csv("ParisHousing.csv")

# California Housing Prices

#full_df = pd.read_csv("train.csv") 

#Selection of features for dataframe

features = ['squareMeters', 'numberOfRooms',  'floors',
       'cityCode', 'cityPartRange', 'numPrevOwners', 'made'
       , 'basement', 'attic', 'garage','price']
df = full_df[[feature for feature in features]]


	
    
# 'Elementary School Distance','Middle School Distance','High School Distance','Appliances included','Annual tax amount','Last Sold Price','City','State'


#handling missing values

def missing_data (dataset):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(dataset)
    dataset = pd.DataFrame(data=imp_mean.transform(dataset), columns=dataset.columns)
    return dataset

#categorical --> numerical

def pre_processing(df):

	""" partioning data into features and target """

	X = df.drop(['price'], axis=1)
	y = df['price']

	return X, y

def normalization(X):
    transformer = preprocessing.MinMaxScaler().fit(X)
    X_transformed = transformer.transform(X)
    return X_transformed

def dummies (training,categorical):
        dummies_df= pd.get_dummies(training, drop_first = True, columns=[var for var in categorical], dtype=float)
        return dummies_df

def segmentation (data,features_list):
	data_segment = data[[feature for feature in features_list]]
	columns = [var for var in data_segment.columns if data_segment[var].dtype=='O']
	return data_segment,columns



dataset,categorical = segmentation(df,features)
numerized_dataset = dummies(dataset,categorical)
dataset = missing_data(numerized_dataset)



X,y = pre_processing(dataset)
X_transform = normalization(X)


sns.heatmap

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size = 0.2, random_state=42)

#%%

fig = px.imshow(full_df[features+['price','hasGuestRoom']].corr(method='spearman').round(2),zmin=-1,zmax=1, text_auto=True,width=1200,height=650,aspect=None,color_continuous_scale=['#DEB078','#003f88'])
fig.show()

#%%

# Linear Regression model

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
linear_pred = reg_all.predict(X_test)


#Metrics and Scoring
    
    # R2 score function 
    
print("Score function for Linear regression model : " + str(reg_all.score(X_test, y_test)*100) + " %")

    # Cross validation score function

cv_results = cross_val_score(reg_all, X_transform, y, cv=5)
print("Cross validation score for Linear regression model : " + str(np.mean(cv_results)*100) + " %")

    # Mean Squared log error function
    
print("MSLE function for Linear regression model : " + str(mean_squared_log_error(y_test, linear_pred)) + "\n" )

#%%

# Ridge Regression model

ridge = make_pipeline(preprocessing.StandardScaler(with_mean=False), Ridge())
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)


#Metrics and Scoring
    
    # R2 score function 
    
print("Score function for Ridge regression model : " + str(ridge.score(X_test, y_test)*100) + " %")

    # Cross validation score function

cv_results = cross_val_score(ridge, X_transform, y, cv=5)
print("Cross validation score for Ridge regression model : " + str(np.mean(cv_results)*100) + " %")

    # Mean Squared log error function
    
print("MSLE function for Ridge regression model : " + str(mean_squared_log_error(y_test, ridge_pred)) + "\n" )

#%%


# Lasso Regression model

lasso = make_pipeline(preprocessing.StandardScaler(with_mean=False), Lasso())
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)


#Metrics and Scoring
    
    # R2 score function 
    
print("Score function for Lasso regression model : " + str(lasso.score(X_test, y_test)*100) + " %")

    # Cross validation score function

cv_results = cross_val_score(lasso, X_transform, y, cv=5)
print("Cross validation score for Lasso regression model : " + str(np.mean(cv_results)*100) + " %")

    # Mean Squared log error function
    
print("MSLE function for Lasso regression model : " + str(mean_squared_log_error(y_test, lasso_pred)) + "\n" )

#%%
# Decision tree Regressor

dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.1,random_state=3)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Metrics and Scoring
    
    # R2 score function 
    
print("Score function for Decision tree regressor model : " + str(dt.score(X_test, y_test)*100) + " %")

    # Cross validation score function

cv_results = cross_val_score(dt, X_transform, y, cv=5)
print("Cross validation score for Decision tree regressor model : " + str(np.mean(cv_results)*100) + " %")

    # Mean Squared log error function
    
print("MSLE function for Decision tree regressor model : " + str(mean_squared_log_error(y_test, dt_pred)) + "\n" )
#%%

# Random forest Regressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
rf_pred = dt.predict(X_test)

# Metrics and Scoring
    
    # R2 score function 
    
print("Score function for Random forest regressor model : " + str(rf.score(X_test, y_test)*100) + " %")

    # Cross validation score function

cv_results = cross_val_score(rf, X_transform, y, cv=5)
print("Cross validation score for Random forest regressor model : " + str(np.mean(cv_results)*100) + " %")

    # Mean Squared log error function
    
print("MSLE function for Random forest regressor model : " + str(mean_squared_log_error(y_test, rf_pred)) + "\n" )
#%%
#https://www.youtube.com/watch?v=V0u6bxQOUJ8