import pandas as pd; import numpy as np; import matplotlib.pyplot as plt

import panel as pn

pn.extension()

data = pd.read_csv('https://raw.githubusercontent.com/holoviz/panel/master/examples/assets/occupancy.csv')
data['date'] = data.date.astype('datetime64[ns]')
data = data.set_index('date')

print(data.tail())
