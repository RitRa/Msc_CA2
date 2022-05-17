import streamlit as st
#streamlit run dashboard.py
import pandas as pd
import numpy as np
import pickle
#data vis

import plotly.express as px



#ml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ml    
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,explained_variance_score
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # Import train_test_split function [2]
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

#setting a wide layout
st.set_page_config(layout="wide")

#columns
header = st.container()
col1, col2 = st.columns(2)
body = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

#fertiliser
df_fertiliser = pd.read_csv("https://raw.githubusercontent.com/RitRa/Msc_CA2/master/data/df_fertiliser.csv", encoding='latin-1')


with header:
    select_type = st.selectbox(
         'Select fertiliser',
         (df_fertiliser['fertiliser_type']))
    st.write('You selected:', select_type)    

with col1:
    st.header("An owl")
    ################## Start of line chart and select box ##################
    
    #select the type of feriliser you are interested in 
    #https://docs.streamlit.io/library/api-reference/widgets/st.selectbox

    # create a dataframe based on selected
    info = df_fertiliser[df_fertiliser['fertiliser_type'] == select_type]
    
   # line plot of the select datae
    fig = px.line(info, x='date',y='value', color='fertiliser_type', title='fertiliser types ')
    
    # disply line Chart
    st.plotly_chart(fig, use_container_width=True)

    ################## End of line chart and select box ##################
    


with col2:
    st.header("A dog")
        
    ################## start of histogram ##################
    
    fig1 = px.histogram(info['value'], title='Histogram of fertiliser types')
    
    # disply line Chart
    st.plotly_chart(fig1, use_container_width=True)
    
    ################## End of histogram ##################
  


    

with body:    


    ################## start of boxplot ##################
    
    # plot of fertiliser type and value
    fig_box = px.box(df_fertiliser, x="fertiliser_type", y="value", title='Boxplot of fertiliser types', color="fertiliser_type" )
    fig_box.update_layout(height=500)
    
    # disply line Chart
    st.plotly_chart(fig_box, use_container_width=True)
    
    ################## End of boxplot ##################
    
    
    ################## start of nbarplot of the number of fertiliser types ##################
    
    df_fertiliser_count= df_fertiliser.dropna()
    #grouping by year and counting the fertiliser types for each year
    df_fertiliser_count = df_fertiliser_count.groupby('year').fertiliser_type.nunique().reset_index()
    #st.write(df_fertiliser_count.head(5))
    
    
    fig_bar_count = px.bar(df_fertiliser_count, x="year", y="fertiliser_type", title='Number of fertiliser types for sale' )
    
    st.plotly_chart(fig_bar_count, use_container_width=True)
    
    ################## End of barplot of the number of fertiliser types ##################
    

   
with dataset:
    st.header("Fertiliser dataset")
    
with features:
    st.header("features")
    
with model_training:
    st.header("models")
    sel_col, disp_col = st.columns(2)
    
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
    
    
    
    
    
    


