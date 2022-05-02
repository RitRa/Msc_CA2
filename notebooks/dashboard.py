import streamlit as st

import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,explained_variance_score
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # Import train_test_split function [2]
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix



header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()



with header:

    st.title("This is a test")
    st.text("this some textx")
    
    df_fertiliser =  pd.read_pickle("https://github.com/RitRa/Msc_CA2/blob/b802b84c1bc267beea543ce96fa8e65f72fb6902/data/df_fertiliser.pkl")
    st.write(df_fertiliser.head(5))
    
    #drop na values
    x = df_fertiliser['value'].dropna()
    
    st.subheader("kdkgdgjdgkdg")
    st.subheader("Weekly Demand Data")

#Bar Chart
    st.bar_chart(x)
    

   
with dataset:
    st.header("Fertiliser dataset")
    
with features:
    st.header("features")
    st.markdown("* **first feature :** sjakhdkjsaghjkdg")
    
with model_training:
    st.header("models")
    sel_col, disp_col = st.columns(2)
    
    """  max_depth = st.slider("What is the max depth?", min_value=10, max_value=100, value = 20, step=20 )
    
    n_esimators = st.selectbox("How many trees should there be?", options=[100,200, 300, 'No limit'], index=0 )
    
    st.text("Here is a list of features in my data:")
    st.write(df_fertiliser.columns)
     """
    # input_feature = st.text_input("which feature should be used as the input feature?")
    
    df_fertiliser = df_fertiliser.dropna()

    #keep date, fertiliser_type, month_year, month, year
    X= df_fertiliser.drop(['fertiliser_type', 'statistic', 'unit'], axis=1) # feature matrix 
    st.write(X.head(5))
    y = df_fertiliser['fertiliser_type']
    class_labels = np.unique(y)
    st.write(y)
    
    #splitting into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # 67% training and 33% test // found out by 1 - test_size = 1 - 0.33 = 0.67 -> 67%
    X_train.shape, X_test.shape


    #encoding
    import category_encoders as ce
    encoder = ce.OrdinalEncoder(cols=['date', 
                                    'month_year', 
                                    'month', 
                                    'year'])

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    
    #scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write(X_test)
    
    
    #label encoding
    label_encoder = LabelEncoder()
    y_train= label_encoder.fit_transform(y_train)
    y_test= label_encoder.fit_transform(y_test)
    st.subheader('y_test:')
    st.write(y_test)


    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    st.subheader('clf:')
    st.write(clf)
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    st.subheader('y_pred:')
    st.write(y_pred)
    
    st.subheader('Accuracy:')
    st.write(classification_report(y_test,  y_pred, target_names=class_labels))
    
    
    
    
    
    


