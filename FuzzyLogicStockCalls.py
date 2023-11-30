#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing Libraries and dependancies
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# In[3]:


# Function to fetch stock data from a custom API
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(tickers = symbol, start = start_date, end = end_date)
    return pd.DataFrame(data)


# In[24]:


# Fetch stock data for a specific symbol and date range
symbol = "BPCL.NS"
start_date = "2023-01-01"
end_date = "2023-11-25"
stock_data = fetch_stock_data(symbol, start_date, end_date)


# In[25]:


# Calculate RSI, EMA12, EMA25 & RVI
stock_data['RSI'] = ta.rsi(stock_data['Close'])
stock_data_EMA25 = ta.ema(stock_data['Close'], length=25)
stock_data['RVI'] = ta.rvi(stock_data['Close'], length=12)

stock_data['EMA25'] = StandardScaler().fit_transform(stock_data_EMA25.values.reshape(-1, 1))
stock_data = stock_data.dropna()


# In[6]:


# Fuzzy logic setup
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'RSI')
ema25 = ctrl.Antecedent(np.arange(0, 1, 0.01), 'EMA25')
rvi = ctrl.Antecedent(np.arange(0, 101, 1), 'RVI')
action = ctrl.Consequent(np.arange(-1, 1.01, 0.01), 'Action')


# In[7]:


# Create fuzzy rule
def fuzzy_rule(rsi_val, ema25_val, rvi_val):
    rsi['low'] = fuzz.trapmf(rsi.universe, [-20, 0, 30, 40])
    rsi['mid'] = fuzz.trapmf(rsi.universe, [20, 30, 70, 80])
    rsi['high'] = fuzz.trapmf(rsi.universe, [40, 70, 80, 100])

    ema25['low'] = fuzz.trapmf(ema25.universe, [0, 0, 0.3, 0.4])
    ema25['mid'] = fuzz.trapmf(ema25.universe, [0.3, 0.4, 0.7, 0.9])
    ema25['high'] = fuzz.trapmf(ema25.universe, [0.5, 0.7, 1.0, 1.0])

    rvi['low'] = fuzz.trapmf(rvi.universe, [0, 0, 30, 50])
    rvi['mid'] = fuzz.trapmf(rvi.universe, [20, 30, 70, 100])
    rvi['high'] = fuzz.trapmf(rvi.universe, [70, 70, 100, 100])

    action['sell'] = fuzz.trimf(action.universe, [-1, -0.3, 0])
    action['hold'] = fuzz.trimf(action.universe, [-0.3, 0, 0.3])
    action['buy'] = fuzz.trimf(action.universe, [0, 0.3, 1])

    rule1 = ctrl.Rule(rsi['low'] & ema25['low'] & rvi['high'], action['sell'])
    rule2 = ctrl.Rule(rsi['mid'] & ema25['mid'] & rvi['mid'], action['hold'])
    rule3 = ctrl.Rule(rsi['high'] & ema25['high'] & rvi['low'], action['buy'])
    
    rule4 = ctrl.Rule(rsi['high'] & ema25['high'] & rvi['high'], action['sell'])
    rule5 = ctrl.Rule(rsi['high'] & ema25['high'] & rvi['mid'], action['sell'])
    rule6 = ctrl.Rule(rsi['mid'] & ema25['mid'] & rvi['high'], action['sell'])
    rule7 = ctrl.Rule(rsi['mid'] & ema25['mid'] & rvi['low'], action['buy'])
    rule8 = ctrl.Rule(rsi['low'] & ema25['low'] & rvi['mid'], action['buy'])
    rule9 = ctrl.Rule(rsi['low'] & ema25['low'] & rvi['low'], action['buy'])
    
    rule10 = ctrl.Rule(rsi['high'] & ema25['mid'] & rvi['high'], action['sell'])
    rule11 = ctrl.Rule(rsi['high'] & ema25['low'] & rvi['high'], action['sell'])
    rule12 = ctrl.Rule(rsi['mid'] & ema25['high'] & rvi['mid'], action['hold'])
    rule13 = ctrl.Rule(rsi['mid'] & ema25['low'] & rvi['mid'], action['hold'])
    rule14 = ctrl.Rule(rsi['low'] & ema25['high'] & rvi['low'], action['buy'])
    rule15 = ctrl.Rule(rsi['low'] & ema25['mid'] & rvi['low'], action['buy'])
    
    rule16 = ctrl.Rule(rsi['mid'] & ema25['high'] & rvi['high'], action['sell'])
    rule17 = ctrl.Rule(rsi['low'] & ema25['high'] & rvi['high'], action['buy'])
    rule18 = ctrl.Rule(rsi['high'] & ema25['mid'] & rvi['mid'], action['sell'])
    rule19 = ctrl.Rule(rsi['low'] & ema25['mid'] & rvi['mid'], action['buy'])
    rule20 = ctrl.Rule(rsi['high'] & ema25['low'] & rvi['low'], action['sell'])
    rule21 = ctrl.Rule(rsi['mid'] & ema25['low'] & rvi['low'], action['buy'])
    
    rule22 = ctrl.Rule(rsi['low'] & ema25['high'] & rvi['mid'], action['buy'])
    rule23 = ctrl.Rule(rsi['low'] & ema25['mid'] & rvi['high'], action['buy'])
    rule24 = ctrl.Rule(rsi['high'] & ema25['mid'] & rvi['low'], action['sell'])
    rule25 = ctrl.Rule(rsi['high'] & ema25['mid'] & rvi['mid'], action['sell'])
    rule26 = ctrl.Rule(rsi['mid'] & ema25['high'] & rvi['low'], action['sell'])
    rule27 = ctrl.Rule(rsi['mid'] & ema25['low'] & rvi['high'], action['buy'])
    
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, 
                                 rule11, rule12, rule13 ,rule14, rule15, rule16, rule17, rule18, rule19,
                                 rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])
    return ctrl.ControlSystemSimulation(system)


# In[26]:


trading_simulator = fuzzy_rule(stock_data['RSI'], stock_data['EMA25'], stock_data['RVI'])

# Trading simulation loop
for i, row in stock_data.iterrows():
    trading_simulator.input['RSI'] = row['RSI']
    trading_simulator.input['EMA25'] = row['EMA25']
    trading_simulator.input['RVI'] = row['RVI']

    # Compute the fuzzy logic
    trading_simulator.compute()
    
    # Make trading decisions based on fuzzy logic output
    if trading_simulator.output['Action'] >= 0.4:
        print(f"Buy {symbol} on {row.name}")
    elif trading_simulator.output['Action'] <= -0.4:
        print(f"Sell {symbol} on {row.name}")


# In[ ]:




