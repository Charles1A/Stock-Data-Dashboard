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
    placeholder = 'Select the relevant index')

analysis = st.sidebar.selectbox(label='Choose analysis to display', options = ['-', 'Multi Percent change', 'Multi Correlation', 'Ratio: Days Up vs Down'])

backdate_52_wks = datetime.now() - timedelta(weeks=52)
api_start_date = backdate_52_wks.strftime('%Y-%m-%d') # reformats start date for yf.download API
UTC_EST_offset = datetime.now() - timedelta(hours=12) # this offset prevents api errors that can occur due to time zone difference with Streamlit's UTC server
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

        yf_stock_data = yf.download(f'{API_index_input} {tickers}', start=api_start_date, end=api_end_date, interval='1d', auto_adjust=False)

        return yf_stock_data

    adj_close_df = yf_api_call(tickers, API_index_input).loc[:, 'Adj Close'] # need only the adjusted close values

    # Below: The return statement creates a tuple of three values:

    return adj_close_df.dropna(axis=0), adj_close_df.index[-1].strftime('%B %d, %Y'), adj_close_df.index[0].strftime('%B %d, %Y')


def pct_change_func(stock_df): # Takes the dataframe returned by stock_data() as an argument

    pct_chg = (stock_df.iloc[-1, :] - stock_df.iloc[-5, :])/stock_df.iloc[-5, :] # Computes percent change between dates

    # The above subsetting operation returns a series; must be converted back to a dataframe

    pct_chg_df = pct_chg.to_frame()

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


def relative_change_func(stock_df): # Takes the dataframe returned by stock_data() as an argument

    pct_chg = (stock_df.iloc[-1, :] - stock_df.iloc[-5, :])/stock_df.iloc[-5, :] # Computes percent change between dates

    # The above subsetting operation returns a series; must be converted back to a dataframe

    pct_chg_df = pct_chg.to_frame()

    pct_chg_df.rename(columns={0: '5d'}, inplace=True)

    pct_chg_df.index.name ='Lookback'

    pct_chg_df['10d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-10, :])/stock_df.iloc[-10, :]

    pct_chg_df['15d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-15, :])/stock_df.iloc[-15, :]

    pct_chg_df['30d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-30, :])/stock_df.iloc[-30, :]

    pct_chg_df['90d'] = (stock_df.iloc[-1, :] - stock_df.iloc[-90, :])/stock_df.iloc[-90, :]

    pct_chg_df['1yr'] = (stock_df.iloc[-1, :] - stock_df.iloc[1, :])/stock_df.iloc[1, :]

    pct_chg_df = pct_chg_df.T # transpose dataframe to re-orient the tickers to the top header row

    if index == 'Nasdaq':

        # Calculate relative returns: subtract index returns column from stock returns columns
        relative_returns = pct_chg_df.sub(pct_chg_df['^IXIC'], axis=0) # axis = 0: horizontal axis

        # Remove the index column from the result
        relative_returns = relative_returns.drop(columns=['^IXIC'])

    if index == 'DJIA':

        # Calculate relative returns by subtracting index returns from stock returns
        relative_returns = pct_chg_df.sub(pct_chg_df['^DJI'], axis=0) # axis = 0: horizontal axis

        # Remove the index column from the result
        relative_returns = relative_returns.drop(columns=['^DJI'])

    if index == 'Russell':

        # Calculate relative returns by subtracting index returns from stock returns
        relative_returns = pct_chg_df.sub(pct_chg_df['^RUT'], axis=0) # axis = 0: horizontal axis

        # Remove the index column from the result
        relative_returns = relative_returns.drop(columns=['^RUT'])


    if index == 'S&P500':

        # Calculate relative returns by subtracting index returns from stock returns
        relative_returns = pct_chg_df.sub(pct_chg_df['^GSPC'], axis=0) # axis = 0: horizontal axis

        # Remove the index column from the result
        relative_returns = relative_returns.drop(columns=['^GSPC'])

    return relative_returns


def correlogram_func(stock_df):

    pct_returns = stock_df.iloc[-lookback_days-1:,:].pct_change() # Computes percent change from the immediately previous row;
    # the number of lookback days is selected in the web app interface; this value determines the number of rows included in the .pct_change computation

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
    # this gets passed as 'mask' argument in the seaborn heatmap function below
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


def up_down_func(stock_df):

    stock_df.reset_index(drop=True, inplace=True)

    stock_df = stock_df.iloc[-lookback_days-1:,:].pct_change() * 100

    stock_df = stock_df.dropna() # remove the first row with no change

    if index == 'Nasdaq':

        stock_df.rename(columns={"^IXIC": "Nasdaq",}, inplace=True)

    if index == 'DJIA':

        stock_df.rename(columns={"^DJI": "DJIA",}, inplace=True)

    if index == 'Russell':

        stock_df.rename(columns={"^RUT": "Russell",}, inplace=True)

    if index == 'S&P500':

        stock_df.rename(columns={"^GSPC": "S&P500",}, inplace=True)

    # Generate a new dataframe that shows 'Up' or 'Down' for each positive change or negative change
    # This will enable us to tabulate the number of days the stock  moved up or down

    up_down_df = stock_df.map(lambda x: "Up" if x > 0 else "Down") # use .map to apply the lambda function elementwise to df
    

    result =[]

    # function that tabulates the number of 'Up' entries and 'Down' entries for each column/ticker:
    for col in up_down_df.columns:

        up_down = up_down_df[col].value_counts()/len(up_down_df) * 100

        result.append(up_down)

    df = pd.concat([pd.DataFrame(l) for l in result], axis=1).T

    df.insert(0, 'ticker', up_down_df.columns)

    df.reset_index(inplace=True)

    df.drop(df.columns[0], axis=1, inplace=True)

    # st.write(df) # un-comment to confirm correct structure of dataframe


    fig, axes = plt.subplots(len(df), 1, figsize=(8, 0.8 * len(df)), sharex=True)
    # fig.suptitle("Percentages of Up and Down Days") # Single title for the entire figure

    for idx, row in df.iterrows():
        up = row['Up']
        down = row['Down']
        ticker = row['ticker']

        ax = axes[idx]
        ax.barh(0, up, color='#487eb0', label='Up')
        ax.barh(0, down, left=up, color='#fa8ca4', label='Down')

        # Annotate bars with percentage values
        ax.annotate(f"{up:.0f}%", xy=(up / 2, 0), va='center', ha='center', color='white', fontweight='bold')
        ax.annotate(f"{down:.0f}%", xy=(up + down / 2, 0), va='center', ha='center', color='white', fontweight='bold')
        
        ax.set_yticks([])  # Remove y-axis ticks and labels
        ax.set_title(f"{ticker}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap, considering the title

    handles, labels = ax.get_legend_handles_labels() # Get handles and labels from the last axes
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    st.pyplot(fig)



# # --- # Start screen

if analysis == '-':

        st.markdown("""

        <h1 style='text-align: center;'>Stock & ETF Comparison Dashboard</h1> 

                """,
        unsafe_allow_html=True) # set to True so that the HTML is rendered rather than just displayed as text

        # Note! change image path based on local or Github location
        st.image('landing_page_banner.png',
            caption="Left: percent price changes; Right: multi-correlation grid")

        st.markdown(
        """
        This dashboard displays several metrics to aid you in evaluating securities side-by-side:

            → Multi Percent change: Price movements expressed as percentages over different time ranges, 
            with option to show % price change or relative % price change

            → Multi Correlation: Price movement correlations over different time ranges

            → Percentage of Days Closed Up and Days Closed Down

        These data show how closely your tickers of interest move together -- sparing you from redundant research and stock buys.

        The analyses displayed here use **adjusted close** prices, which factor in stock splits and dividends. 
        
        <footer style='text-align: left;'>
        Notes: <br>  
        This app uses a free API; price data may be delayed. <br>
        The app might break from time to time due to API updates. Please report any malfunction. <br>
        </footer>
        """,
        unsafe_allow_html=True)

# # --- # Percent changes

if analysis == 'Multi Percent change' and len(ticker_list) >= 2 and len(ticker_list) <= 4 and index != '-':

    tab1, tab2 = st.tabs(["Pct Changes", "Performance Relative to Index"])

    with tab1:

        st.markdown("""

            <h1 style='text-align: center;'>Price Change Comparison</h1> 

                    """,
            unsafe_allow_html=True)

        # below: styling parameters that are fed into the "set table styles" function of Pandas
        # each table_style is a dictionary with 'selector' and 'props keys'
        # each 'selector' is a CSS selector; each 'props' follows the format (attribute, value)

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

        st.info(f"**'d'** denotes trading days counted backward from {d1}, the most-recent trading date for which data can be retrieved.")

        st.write(f'{table}', unsafe_allow_html=True)

    with tab2:

        st.markdown("""

        <h1 style='text-align: center;'>Performance Relative to Index</h1>

        <h6 style='text-align: center;'>See whether your securities overperformed or underperformed the index</h6>

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

        table = relative_change_func(stock_data()[0]).style.format('{:.1%}') \
        .bar(height=70, width=50, align='mid', vmin=-0.6, vmax=0.6, cmap='RdBu') \
        .set_table_attributes('style="border-collapse:collapse"') \
        .set_table_styles([s1, s2, s3, s5, s6, s7, s8]) \
        .to_html()

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

        st.write(f"The selected lookback period spans {lookback_date} to {stock_data()[1]} (the most-recent trading date for which data were retreived)")

    with col2:

        st.write("")

    with col3:

        correlogram_func(stock_data()[0])


if analysis == 'Ratio: Days Up vs Down' and len(ticker_list) >= 2 and len(ticker_list) <= 4 and index != '-':

    st.markdown("""

    <h1 style='text-align: center;'>Percentage of Days Closed Up and Days Closed Down</h1> 

            """,
    unsafe_allow_html=True)

    lookback_days = st.slider(label = 'Use the slider to select the number of elapsed trading days to include', 
    min_value=10, 
    max_value=170, 
    value=90, 
    step=10)

    lookback_date = stock_data()[0].index[-lookback_days].strftime('%B %d, %Y')

    st.write(f"The selected lookback period spans {lookback_date} to {stock_data()[1]} (the most-recent trading date for which data were retreived)")

    up_down_func(stock_data()[0])
























