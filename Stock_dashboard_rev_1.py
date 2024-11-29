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

st.set_page_config(page_title="Stock Data Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout='wide', 
    initial_sidebar_state='auto')
 

# # --- # Sidebar elements

tickers = st.sidebar.text_input(label = 'Enter 2 to 4 tickers, each separated with a space (e.g., \'nvda TSLA Arkk\')')
tickers = tickers.upper() # make lower-case letters upper-case
ticker_list = tickers.split(" ") # split tickers on single space to create list of tickers

if len(ticker_list) == 1 or len(ticker_list) > 4:

    st.sidebar.warning('Enter at least 2 but no more than 4 tickers')


index_tuple = ('-', 'Nasdaq', 'DJIA', 'Russell', 'S&P500') # Tuple of benchmarks for the selectbox
index = st.sidebar.selectbox(label='Select a benchmark index for comparison',
    options = index_tuple, 
    placeholder = 'Select a relevant index')

analysis = st.sidebar.selectbox(label='Choose analysis to display', options = ['-', 'Multi Percent change', 'Multi Correlation'])

backdate_52_wks = datetime.now() - timedelta(weeks=52)
api_start_date = backdate_52_wks.strftime('%Y-%m-%d') # reformats start date for yf.download API
UTC_EST_offset = datetime.now() - timedelta(hours=5) # this is to prevent api errors that occur due to Streamlit's UTC server
api_end_date = UTC_EST_offset.strftime('%Y-%m-%d') # reformats end date for yf.download API

# # --- # Function definitions

def stock_data():

    if index == 'Nasdaq':
        API_index_input = '^IXIC' # Ticker for Nasdaq Composite

    if index == 'DJIA':
        API_index_input = '^DJI' # Ticker for Dow Jones Industrial Average

    if index == 'Russell':
        API_index_input = '^RUT' # Ticker for Russell 2000 Index

    if index == 'S&P500':
        API_index_input = '^GSPC' # Ticker for Standard and Poor's 500 Index

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

        yf_stock_data = yf.download(f'{API_index_input} {tickers}', start=api_start_date, end=api_end_date, interval='1d')

        return yf_stock_data

    adj_close_df = yf_api_call(tickers, API_index_input).loc[:, 'Adj Close'] # need only the adjusted close values

    # Below: The return statement creates a tuple of three values:

    return adj_close_df.dropna(axis=0), adj_close_df.index[-1].strftime('%B %d, %Y'), adj_close_df.index[0].strftime('%B %d, %Y')


def pct_change_func(stock_df): # Takes the dataframe returned by stock_data() as an argument

    pct_chg_df = (stock_df.iloc[-1, :] - stock_df.iloc[-5, :])/stock_df.iloc[-5, :] # Computes percent change between dates

    # The above subsetting operation returns a series; must be converted back to a dataframe

    pct_chg_df = pct_chg_df.to_frame()

    pct_chg_df.rename(columns={0: '5d'}, inplace=True)

    pct_chg_df.index.name ='Lookback'

    pct_chg_df['10d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-10, :])/stock_df.iloc[-10, :]

    pct_chg_df['15d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-15, :])/stock_df.iloc[-15, :]

    pct_chg_df['30d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-30, :])/stock_df.iloc[-30, :]

    pct_chg_df['90d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-90, :])/stock_df.iloc[-90, :]

    pct_chg_df['1yr'] = (stock_df.iloc[-1, :] - stock_df.iloc[1, :])/stock_df.iloc[1, :]

    pct_chg_df = pct_chg_df.T # transpose dataframe to re-orient the tickers to the top header row

    if index == 'Nasdaq':

        pct_chg_df.rename(columns={"^IXIC": "Nasdaq",}, inplace=True)

        pct_chg_df.insert(0, 'Nasdaq', pct_chg_df.pop('Nasdaq')) # move the benchmark index to the first column

    if index == 'DJIA':

        pct_chg_df.rename(columns={"^DJI": "DJIA",}, inplace=True)

        pct_chg_df.insert(0, 'DJIA', pct_chg_df.pop('DJIA'))

    if index == 'Russell':

        pct_chg_df.rename(columns={"^RUT": "Russell",}, inplace=True)

        pct_chg_df.insert(0, 'Russell', pct_chg_df.pop('Russell'))

    if index == 'S&P500':

        pct_chg_df.rename(columns={"^GSPC": "S&P500",}, inplace=True)

        pct_chg_df.insert(0, 'S&P500', pct_chg_df.pop('S&P500'))

    return pct_chg_df


def correlogram_func(stock_df):

    pct_returns = stock_df.iloc[-lookback_days-1:,:].pct_change() # Computes percent change from the immediately previous row

    if index == 'Nasdaq':

        pct_returns.rename(columns={"^IXIC": "Nasdaq",}, inplace=True)

    if index == 'DJIA':

        pct_returns.rename(columns={"^DJI": "DJIA",}, inplace=True)

    if index == 'Russell':

        pct_returns.rename(columns={"^RUT": "Russell",}, inplace=True)

    if index == 'S&P500':

        pct_returns.rename(columns={"^GSPC": "S&P500",}, inplace=True)

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

        <h1 style='text-align: center;'>Stock & ETF Comparison Dashboard</h1> 

                """,
        unsafe_allow_html=True)

        st.image('/Users/cea/Desktop/Coding and Machine Learning/Python/Dashboards/landing_page_banner.png',
            caption="Left: relative price changes; Right: multi-correlation grid")

        st.markdown(
        """
        This dashboard displays two metrics to aid you in evaluating securities side-by-side:

        • **Multi Percent change**: Price movements expressed as percentages over different time ranges

        • **Multi Correlation**: Price movement correlations over different time ranges

        These data show how closely your tickers of interest move together -- potentially sparing you from redundant research and stock buys.

        The analyses displayed here use **adjusted close** prices, which factor in stock splits and dividends. 
        
        *Note: Data may be delayed up to one day*

        """
        )

# # --- # Percent changes

if analysis == 'Multi Percent change' and len(ticker_list) >= 2 and len(ticker_list) <= 4 and index != '-':

    st.markdown("""

        <h1 style='text-align: center;'>Stock & ETF Price Change Comparison</h1> 

                """,
        unsafe_allow_html=True)

    s8 = dict(selector='th.index_name', \
        props=[('color', 'black'), ('background-color', 'white'), \
        ('font-size', '0.8em'), ('font-weight','bold')])

    s7 = dict(selector='th.col_heading', \
        props=[('text-align', 'center'), ('font-size', '1em'), \
        ('color', 'white'), ('background-color', '#40739e'), ('border-left', '2px solid white')])

    s6 = dict(selector='th:not(.col_heading)', \
        props=[('color', 'white'), ('background-color', '#487eb0'), \
        ('text-align', 'right')])

    s5 = dict(selector='td:nth-child(2)', \
        props=[('border-left', '2px solid white')])

    s3 = dict(selector='td', \
        props=[('border-left', '2px solid #40739e')])

    s2 = dict(selector='th', \
        props=[('border', 'none')])

    # align cell text
    s1 = dict(selector='td', \
        props=[('text-align', 'right')])

    d1 = stock_data()[1]

    table = pct_change_func(stock_data()[0]).style.format('{:.1%}') \
    .bar(height=70, width=50, align='mid', vmin=-0.6, vmax=0.6, cmap='RdBu') \
    .set_table_attributes('style="border-collapse:collapse"') \
    .set_table_styles([s1, s2, s3, s5, s6, s7, s8]) \
    .to_html()

    st.info(f"Below, **d** denotes trading days counted backward from {d1}. This is the most-recent trading date with available data for your tickers.")

    st.write(f'{table}', unsafe_allow_html=True)

# # --- # Correlation matrix

if analysis == 'Multi Correlation' and len(ticker_list) >= 2 and len(ticker_list) <= 4 and index != '-':


    st.markdown("""

        <h1 style='text-align: center;'>Stock Price Correlations</h1> 

                """,
        unsafe_allow_html=True)

    col1, col2, col3= st.columns((1, 0.2, 1))


    with col1:

        st.markdown(
            ''' 
            ► The grid shows **Pearson's R** correlation values

            ► These values signify how strongly two securities are correlated (**Max = 1**)

            '''
            )

        lookback_days = st.slider(label = 'Use the slider to select the number of elapsed trading days to include in correlation computation', 
        min_value=10, 
        max_value=170, 
        value=90, 
        step=10)

        lookback_date = stock_data()[0].index[-lookback_days].strftime('%B %d, %Y')

        st.write(f"The selected lookback period spans {lookback_date} to {stock_data()[1]} (the most-recent closed trading day with uploaded data)")

    with col2:

        st.write("")

    with col3:

        correlogram_func(stock_data()[0])


















