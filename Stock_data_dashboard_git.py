# # --- # Import statements

import yfinance as yf

import scipy
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import time
import re
import matplotlib as mpl

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
 

# # --- # Sidebar elements

tickers = st.sidebar.text_input(label = 'Enter 2 to 4 tickers, each separated with a space (e.g., \'nvda TSLA Arkk\')')
tickers = tickers.upper() # make lower-case letters upper-case
ticker_list = tickers.split(" ") # split tickers on single space to create list of tickers

if len(ticker_list) == 1 or len(ticker_list) > 4:

    st.sidebar.warning('Enter at least 2 but no more than 4 tickers')


index_tuple = ('-', 'Nasdaq', 'Dow', 'Russell') # Tuple of benchmarks for the selectbox
index = st.sidebar.selectbox(label='Select a benchmark index for comparison',
    options = index_tuple, 
    placeholder = 'Select a relevant index')

analysis = st.sidebar.selectbox(label='Choose analysis to display', options = ['-', 'Multi Percent change', 'Multi Correlation'])

# Start date and end date arguments for yf.download API:
days_ago_270 = datetime.now() - timedelta(days=270) # computes date 270 calendar days ago
api_start_date = days_ago_270.strftime('%Y-%m-%d') # formats start date for yf.download API
api_end_date = date.today().strftime('%Y-%m-%d') # formats end date for yf.download API

# # --- # Function definitions

def stock_data():

    if index == 'Nasdaq':
        API_index_input = '^IXIC' # Ticker for Nasdaq Composite

    if index == 'Dow':
        API_index_input = '^DJI' # Ticker for Dow Jones Industrial Average

    if index == 'Russell':
        API_index_input = '^RUT' # Ticker for Russell 2000 Index

    @st.cache_data
    # When the yf_api_call function, below, is called for the first time, 
    # the return value will be stored in the cache via @st.cache_data.
    # This means the function will not be re-executed unless the parameters 
    # ('tickers' or 'index_input') change. 
    # This pre-empts redundant calls to the yf.download API.
    # yf_api_call func takes two arguments: 
    # tickers, a global variable, defined outside the func; 
    # and API_index_input, a local variable
    def yf_api_call(tickers, API_index_input): 

        yf_stock_data = yf.download(f'{API_index_input} {tickers}', start=api_start_date, end=api_end_date)

        return yf_stock_data

    adj_close_df = yf_api_call(tickers, API_index_input).loc[:, 'Adj Close'] # need only the adjusted close values

    # The return statement below creates a tuple of four values: 
    # (1) the adjusted close dataframe transposed and (2) untransposed 
    # (3) the calendar date 180 trading days ago; 
    # (4) the calendar date of the last trading day
    return adj_close_df.T, adj_close_df, adj_close_df.index[-181].strftime('%B %d, %Y'), adj_close_df.index[-1].strftime('%B %d, %Y')

def pct_change_func(stock_df): # Takes the dataframe returned by stock_data() as an argument

    pct_chg_df = (stock_df.iloc[:, -1] - stock_df.iloc[:, -6])/stock_df.iloc[:, -6] # slices data for the last 5 trading days

    # The above subsetting operation returns a series; need to convert back to a dataframe

    pct_chg_df = pct_chg_df.to_frame()

    pct_chg_df.rename(columns={0: '5d'}, inplace=True)

    pct_chg_df.index.name ='Lookback Period (Trading Days)'

    pct_chg_df['15d'] = (stock_df.iloc[:,-1] - stock_df.iloc[:,-16])/stock_df.iloc[:,-16]# Computes percent change


    pct_chg_df['25d'] = (stock_df.iloc[:,-1] - stock_df.iloc[:,-26])/stock_df.iloc[:,-26]# Computes percent change


    pct_chg_df['45d'] = (stock_df.iloc[:,-1] - stock_df.iloc[:,-46])/stock_df.iloc[:,-46]# Computes percent change


    pct_chg_df['90d'] = (stock_df.iloc[:,-1] - stock_df.iloc[:,-91])/stock_df.iloc[:,-91]# Computes percent change


    pct_chg_df['120d'] = (stock_df.iloc[:,-1] - stock_df.iloc[:,-121])/stock_df.iloc[:,-121]# Computes percent change


    pct_chg_df = pct_chg_df.T

    if index == 'Nasdaq':

        pct_chg_df.rename(columns={"^IXIC": "Nasdaq",}, inplace=True)

        pct_chg_df.insert(0, 'Nasdaq', pct_chg_df.pop('Nasdaq')) # move the benchmark index to the first column

    if index == 'Dow':

        pct_chg_df.rename(columns={"^DJI": "Dow",}, inplace=True)

        pct_chg_df.insert(0, 'Dow', pct_chg_df.pop('Dow'))

    if index == 'Russell':

        pct_chg_df.rename(columns={"^^RUT": "Russell",}, inplace=True)

        pct_chg_df.insert(0, 'Russell', pct_chg_df.pop('Russell'))

    return pct_chg_df

def correlogram_func(stock_df):

    pct_returns = stock_df.iloc[-lookback_days:,:].pct_change() # Computes percent change from the immediately previous row

    if index == 'Nasdaq':

        pct_returns.rename(columns={"^IXIC": "Nasdaq",}, inplace=True)

    if index == 'Dow':

        pct_returns.rename(columns={"^DJI": "Dow",}, inplace=True)

    if index == 'Russell':

        pct_returns.rename(columns={"^^RUT": "Russell",}, inplace=True)

    corr_matrix = pct_returns.corr() # computes pairwise correlation of columns

    mask = np.zeros_like(corr_matrix) # returns an array of zeros with the same shape as input array

    triangle_indices = np.triu_indices_from(mask) # return the indices for the upper-triangle of mask

    mask[triangle_indices] = True # Sets the upper-triangle indices to '1' (True)
    # this gets passed as 'mask' argument in seaborn heatmap function below
    # data will not be shown in cells where mask is True
    # prevents same values from appearing twice in heatmap

    fig, ax = plt.subplots(figsize=(3,3))

    ax = sns.heatmap(ax=ax, 
        data=corr_matrix, 
                 mask=mask,
                 annot=True, 
                 linewidth=.5, cmap="crest", 
                 annot_kws={"size": 7})

    ax.set_title(f'Correlogram') # optional title;

    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.tick_params(axis='y', labelrotation=0, labelsize=7)

    return st.pyplot(fig)

# # --- # Start screen

if analysis == '-':

        st.markdown("""

        <h1 style='text-align: center;'>Stock Data Dashboard</h1> 

                """,
        unsafe_allow_html=True)

        st.markdown(
        """
        This dashboard displays two metrics to aid you in evaluating **up to four equities** side-by-side!

        • Price movements as percentages over different time frames

        • Price movement correlations over different time frames

        ##### To get started: provide the necessary inputs on the left.

        """
        )

# # --- # Percent changes

if analysis == 'Multi Percent change' and len(ticker_list) >= 2 and len(ticker_list) <= 4 and index != '-':

    st.markdown("""

        <h1 style='text-align: center;'>Equity Price Changes</h1> 

                """,
        unsafe_allow_html=True)

    s9 = dict(selector='caption', \
        props=[('text-align', 'center'), ('caption-side', 'top'), \
        ('color', 'black'), ('font-size', '1em')])

    s8 = dict(selector='th.index_name', \
        props=[('color', 'black'), ('background-color', 'white'), \
        ('font-size', '0.8em'), ('font-weight','bold')])

    s7 = dict(selector='th.col_heading', \
        props=[('text-align', 'center'), ('font-size', '1.2em'), \
        ('color', 'white'), ('background-color', '#40739e')])

    s6 = dict(selector='th:not(.col_heading)', \
        props=[('color', 'white'), ('background-color', '#487eb0'), \
        ('text-align', 'right')])

    s5 = dict(selector='td:nth-child(2)', \
        props=[('border-left', '2px solid white')])

    s4 = dict(selector='th:nth-child(3)', \
        props=[('border-left', '2px solid white')])

    s3 = dict(selector='td:nth-child(3)', \
        props=[('border-left', '2px solid black')])

    s2 = dict(selector='th, td', \
        props=[('border', 'none')])

    s1 = dict(selector='td', \
        props=[('text-align', 'center')])

    table = pct_change_func(stock_data()[0]).style.format('{:.1%}') \
    .bar(align=0, vmin=-0.4, vmax=0.4, cmap='RdBu') \
    .set_caption(f"Lookback period counts backward from most-recent U.S. market trading day: {stock_data()[3]}") \
    .set_table_attributes('style="border-collapse:collapse"') \
    .set_table_styles([s1, s2, s3, s4, s5, s6, s7, s8, s9]) \
    .to_html()
    
    st.write(f'{table}', unsafe_allow_html=True)


# # --- # Correlation matrix

if analysis == 'Multi Correlation' and len(ticker_list) >= 2 and len(ticker_list) <= 4 and index != '-':


    st.markdown("""

        <h1 style='text-align: center;'>Pearson\'s R correlations</h1> 

                """,
        unsafe_allow_html=True)

    col1, col2, col3= st.columns((1, 0.2, 1))


    with col1:


        st.markdown(
            ''' 
            ► **R** signifies how strongly two variables are correlated and in which direction (negative or positive)

            ► **R** values range from -1 to 1

            '''
            )

        lookback_days = st.slider(label = 'Lookback period: Select number of trading days to include in correlation computation', 
        min_value=5, 
        max_value=180, 
        value=90, 
        step=15)

        start_date = stock_data()[1].index[-lookback_days].strftime('%B %d, %Y')

        st.write(f"The selected lookback period covers {start_date} to {stock_data()[3]}")

    with col2:

        st.write("")

    with col3:

        correlogram_func(stock_data()[1])






