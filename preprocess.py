# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns



# Paris Housing Prices

full_df = pd.read_csv("datasets/ParisHousing.csv")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

if __name__ == "__main__":
    
    print("Données traitées")
    
    

