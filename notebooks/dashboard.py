import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
import pickle
import csv

# Import the os module
import os

# Get the current working directory
#cwd = os.getcwd()
#print(cwd)

#os.chdir(r"/Users/ritaraher/Dropbox/Study/CCT/python_msc/Msc_CA2/notebooks")
#print("Directory changed")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#df_fertiliser = pd.read_pickle("../data/df_fertiliser.pkl")
#df_tweets= pd.read_csv("../data/farmtweets.csv")

#df = pd.read_csv('/Users/ritaraher/Dropbox/Study/CCT/python_msc/Msc_CA2/notebooks/restaurants_zomato.csv',encoding="ISO-8859-1")
data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':[20, 21, 19, 18]}
 
# Create DataFrame
df = pd.DataFrame(data)

print("Hello world!")

