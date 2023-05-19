# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

from preprocess import X_train, X_test, y_train, y_test,X_transform,y

def model(X):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr.predict(X)

#AJOUTER LE CHOIX DE MODELE
if __name__=="__main__":
    
    print("Comparatif des modèles")
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

