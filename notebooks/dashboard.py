import streamlit as st
from streamlit_option_menu import option_menu
#streamlit run dashboard.py
import pandas as pd
import numpy as np
import pickle
#data vis

import plotly.express as px

from dateutil.relativedelta import relativedelta
import math

import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import statsmodels.api as sm

#ml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ml    
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,explained_variance_score
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # Import train_test_split function [2]
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix



#setting a wide layout
st.set_page_config(page_title="Farming prices", layout="wide")

#columns
header = st.container()
col1, col2 = st.columns(2)
page2 = st.container()
dataset = st.container()
features = st.container()


#fertiliser
df_fertiliser = pd.read_csv("https://raw.githubusercontent.com/RitRa/Msc_CA2/master/data/df_fertiliser.csv", encoding='latin-1')

df_all = pd.read_csv("https://raw.githubusercontent.com/RitRa/Msc_CA2/master/data/df_all.csv", encoding='latin-1')



# Using "with" notation
with st.sidebar:
    selected = option_menu("Main Menu", ['Overview', 'Models',], 
        icons=['', ''], menu_icon="cast", default_index=1)
    #selected
    
if selected == 'Overview' :
     
    with header:
        st.header("Overview of fertiliser prices")
        ################## Start of scatter plot animation ##################
        xmin, xmax = min(df_fertiliser["value"]), max(df_fertiliser["value"])
        
        fig_test = px.scatter(df_fertiliser,  x="value", y="value",
                            animation_frame="year", animation_group="year", 
                            color="fertiliser_type", hover_name="fertiliser_type", 
                            range_x= [xmin, xmax], range_y=[xmin, xmax]
                    )
        
        st.plotly_chart(fig_test, use_container_width=True)
        ################## End of scatter plot animation ##################
        
        
        ################## start of boxplot ##################
    
        # plot of fertiliser type and value
        fig_box = px.box(df_fertiliser, x="fertiliser_type", y="value", title='Range of pricing for each fertiliser type', color="fertiliser_type" )
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
    
        select_type = st.selectbox(
                'Select fertiliser type',
                (df_fertiliser['fertiliser_type']))
        st.write('You selected:', select_type)  
            
    with col1:

        ################## Start of line chart and select box ##################
        
        #select the type of feriliser you are interested in 
        #https://docs.streamlit.io/library/api-reference/widgets/st.selectbox

        # create a dataframe based on selected
        info = df_fertiliser[df_fertiliser['fertiliser_type'] == select_type]
            
        # col1.metric("Fertiliser price", most_recent_price, max_date)
        
    # line plot of the select datae
        fig = px.line(info, x='date', y='value', color='fertiliser_type', title='Fertiliser types')
        
        # disply line Chart
        st.plotly_chart(fig, use_container_width=True)

        ################## End of line chart and select box ##################
    
    with col2:
        
        ################## start of histogram ##################
        fig1 = px.histogram(info['value'], title='Histogram of fertiliser types')
        
        # disply line Chart
        st.plotly_chart(fig1, use_container_width=True)
        ################## End of histogram ##################


##################  Page 2 ##################  
elif selected == 'Models' :
    with page2:
        
        


        sel_col, disp_col = st.columns(2)
        
        # independant and dependant variables
        X_new= df_all.drop(['fertiliser_price', 'month_year', 'Unnamed: 0'], axis=1) # feature matrix 
        feature_name = X_new.columns
        y_new = df_all['fertiliser_price']
        class_labels = np.unique(y_new)
        

        # columns to encode 
        encoder = ce.OrdinalEncoder(cols=[ 
                                        'fertiliser_type',
                                        'month', 
                                        'year', 'date'])

        X_new = encoder.fit_transform(X_new)
        
        #columns to one hot encode
        ct = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(categories='auto'), [0, 1, 2, 3, 4])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
            remainder='passthrough'                                         # Leave the rest of the columns untouched
        )

        X_new =ct.fit_transform(X_new)
        
        #splitting data
        from sklearn.model_selection import train_test_split # Import train_test_split function [2]
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=0) # 67% training and 33% test // found out by 1 - test_size = 1 - 0.33 = 0.67 -> 67%
            
        #scaling
        sc = StandardScaler(with_mean=False)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        
        
    #################### start select a model and show a plot###############################
        with header: 
            models = {'Linear': LinearRegression,
                        'Lasso': Lasso,
                        'Ridge': Ridge, 
                        'ElasticNet': ElasticNet} 
                    
            st.subheader("Predicted the price of feriliser")
            select_model = st.selectbox(
                            'Select model:',
                            ("Linear", "Lasso", "Ridge", "ElasticNet"))
                    
            st.write('You selected:', select_model)  
        
        
        # function that takes in the select model, list of models, test and training data
        def plotmodel(select_model, models, X_train, y_train, X_test, y_test):
            for key, value in models.items():
                    #st.write(i)
                if select_model == key:
                        
                    # model
                    model = value()
                    model.fit(X_train, y_train)
                    #Predict the response for test dataset
                    y_pred = model.predict(X_test)
                    with col1:
                    # Dataframe of results
                        model_result=pd.DataFrame({ 'Actual':y_test, 'Predicted':y_pred, 'Difference': (y_test - y_pred)})
                        st.write(model_result.head(5))
                    with col2:
                        mae = round(mean_absolute_error(y_test,y_pred), 2)
                        st.write("mean_absolute_error", mae)
                        mse = round(mean_squared_error(y_test,y_pred), 2)
                        st.write("mean_squared_error", mse)
                        model_rmse = round(math.sqrt(mse), 2)
                        st.write("Root mean squared error", model_rmse)
                        r2 = round(r2_score(y_test,y_pred), 2)
                        st.write("R2 score:", r2)
                        precision = round(model.score(X_train, y_train), 2)
                        st.write("Training model precision:", precision)
                    
                
                    
        
                    #plot actual vs predicted and ols trendline
                    fig_model = px.scatter(model_result, x='Actual', y='Predicted', opacity=0.65,  trendline="ols", title="Actual Vs Predicted using %s" % (key),  trendline_color_override='darkblue')
                    
                    #plot on streamlit
                    st.plotly_chart(fig_model, use_container_width=True)
                        
            
        plotmodel(select_model, models, X_train, y_train, X_test, y_test)
        

    #################### End select a model and show a plot###############################
        
        
        
        


