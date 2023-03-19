import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

full_df = pd.read_csv("train.csv")

#Selection of features for dataframe
#%%
features = ['Id','Sold Price','Type','Year built','Heating','Cooling','Parking','Bedrooms','Bathrooms','Total spaces','Garage spaces','Region']
df = full_df[[feature for feature in features]]


	
    
# 'Elementary School Distance','Middle School Distance','High School Distance','Appliances included','Annual tax amount','Last Sold Price','City','State'
#%%

#handling missing values

def missing_data (dataset):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(dataset)
    dataset = pd.DataFrame(data=imp_mean.transform(dataset), columns=dataset.columns)
    return dataset
#catagorical pre processing phase 

'''
for col_name in features:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print ("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
        print(df[col_name].value_counts().sort_values(ascending=False).head())
'''

#categorical --> numerical

def pre_processing(df):

	""" partioning data into features and target """

	X = df.drop(['Sold Price'], axis=1)
	y = df['Sold Price']

	return X, y

def dummies (training,categorical):
        normalized_df= pd.get_dummies(training, drop_first = True, columns=[var for var in categorical], dtype=float)
        return normalized_df

def segmentation (data,features_list):
	data_segment = data[[feature for feature in features_list]]
	columns = [var for var in data_segment.columns if data_segment[var].dtype=='O']
	return data_segment,columns



dataset,categorical = segmentation(df,features)
numerized_dataset = dummies(dataset,categorical)
dataset = missing_data(numerized_dataset)
print(dataset.isnull().sum().sort_values(ascending=False).head())
X,y = pre_processing(dataset)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



#https://www.youtube.com/watch?v=V0u6bxQOUJ8