import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import seaborn as sns
#import pandas_datareader.data as web
from pmdarima.arima import auto_arima
import datetime
import sys


df = pd.read_csv("~/Desktop/Springboard_Capstone3/data/train.csv")
df.Date=pd.to_datetime(df.Date, format='%Y-%m-%d')
sdf = pd.read_csv("~/Desktop/Springboard_Capstone3/data/combined_data.csv")

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error


# from sklearn.metrics import r2_score


class modeling:
    warnings.filterwarnings("ignore")

    def __init__(self, data, storeid):
        self.data = data
        self.storeid = storeid
        self.storeweeklysales = self.data.loc[self.data.Store == self.storeid].set_index('Date').resample(
            'w').Sales.sum()
        self.average = np.full((8,), self.storeweeklysales[:-9].mean())
        self.mae = mean_absolute_error(self.storeweeklysales[-9:-1], self.average)
        self.WAPE = np.sum(abs(self.average - self.storeweeklysales[-9:-1])) / (
            self.data.set_index('Date').resample('w').Sales.sum()[-9:-1].sum())

    def predict(self):
        # storeweeklysales = self.data.loc[self.data.Store == self.storeid].set_index('Date').resample('w').Sales.sum()

        if len(self.storeweeklysales[(self.storeweeklysales == 0)].index) == 0:
            train, test = (self.storeweeklysales[:-9], self.storeweeklysales[-9:-1])
            arima_model = auto_arima(train, start_p=0, d=0, start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            predictresult1 = arima_model.predict(n_periods=8)
            true_values = test.values
            mae = mean_absolute_error(true_values, predictresult1)
            WAPE = np.sum(abs(predictresult1 - true_values)) / (
                self.data.set_index('Date').resample('w').Sales.sum()[-9:-1].sum())

            # R2 = r2_score(true_values, predictresult1)
            return pd.DataFrame([mae, WAPE, self.mae, self.WAPE],
                                index=['MAE_AutoArima', 'WAPE_AutoArima', 'MAE_MeanMethod', 'WAPE_MeanMethod'],
                                columns=[self.storeid]).T
            # return pd.DataFrame([mae, R2])

        if len(self.storeweeklysales.loc[
               self.storeweeklysales[(self.storeweeklysales == 0)].index[-1]:].index) >= 2 and len(
                self.storeweeklysales.loc[
                self.storeweeklysales.loc[self.storeweeklysales[(self.storeweeklysales == 0)].index[-1]:].index[1]:][
                :-1]) >= 16:
            train_startdate = \
            self.storeweeklysales.loc[self.storeweeklysales[(self.storeweeklysales == 0)].index[-1]:].index[1]
            train_enddate = len(self.storeweeklysales.loc[train_startdate:]) - 9
            train = self.storeweeklysales.loc[train_startdate:][:train_enddate]
            # test = self.storeweeklysales.loc[train_startdate:][train_enddate:-1]
            test = self.storeweeklysales[-9:-1]

            arima_model = auto_arima(train, start_p=0, d=0,
                                     start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            predictresult = arima_model.predict(n_periods=8)
            true_values = test.values
            mae_2 = mean_absolute_error(true_values, predictresult)
            WAPE_2 = np.sum(abs(predictresult - true_values)) / (
                self.data.set_index('Date').resample('w').Sales.sum()[-9:-1].sum())
            # R2_2 = r2_score(true_values, predictresult)
            return pd.DataFrame([mae_2, WAPE_2, self.mae, self.WAPE],
                                index=['MAE_AutoArima', 'WAPE_AutoArima', 'MAE_MeanMethod', 'WAPE_MeanMethod'],
                                columns=[self.storeid]).T
            # return pd.DataFrame([mae_2, R2_2])

        else:
            storeweeklysales_dropzero = self.storeweeklysales[self.storeweeklysales != 0]
            train, test = (storeweeklysales_dropzero[:-9], storeweeklysales_dropzero[-9:-1])
            arima_model = auto_arima(train, start_p=0, d=0, start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            predictresult1 = arima_model.predict(n_periods=8)
            true_values = test.values
            mae = mean_absolute_error(true_values, predictresult1)
            WAPE = np.sum(abs(predictresult1 - true_values)) / (
                self.data.set_index('Date').resample('w').Sales.sum()[-9:-1].sum())
            # R2 = r2_score(true_values, predictresult1)
            return pd.DataFrame([mae, WAPE, self.mae, self.WAPE],
                                index=['MAE_AutoArima', 'WAPE_AutoArima', 'MAE_MeanMethod', 'WAPE_MeanMethod'],
                                columns=[self.storeid]).T
            # return pd.DataFrame([mae, R2])

    def forecast(self, start_date, end_date):
        # storeweeklysales = self.data.loc[self.data.Store == self.storeid].set_index('Date').resample('w').Sales.sum()
        # storeweeklysales = storeweeklysales[storeweeklysales!=0]
        forecastindex = pd.date_range(start_date, end_date, freq='W')
        if len(self.storeweeklysales[(self.storeweeklysales == 0)].index) == 0:
            train, test = (self.storeweeklysales[:-9], self.storeweeklysales[-9:-1])
            arima_model = auto_arima(self.storeweeklysales[:test.index[-1]], start_p=0, d=0, start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            forecast, CI = arima_model.predict(n_periods=8, return_conf_int=True, alpha=0.05)
            forecastresult1 = pd.DataFrame([forecast, CI[:, 0], CI[:, 1]],
                                           index=['Forecast', 'lower CI', 'upper CI']).T.set_index(forecastindex)
            forecastresult1['Previous 8 Weeks'] = self.storeweeklysales[-9:-1].values
            return forecastresult1

        if len(self.storeweeklysales.loc[
               self.storeweeklysales[(self.storeweeklysales == 0)].index[-1]:].index) >= 2 and len(
                self.storeweeklysales.loc[
                self.storeweeklysales.loc[self.storeweeklysales[(self.storeweeklysales == 0)].index[-1]:].index[1]:][
                :-1]) >= 16:
            train_startdate = \
            self.storeweeklysales.loc[self.storeweeklysales[(self.storeweeklysales == 0)].index[-1]:].index[1]
            train_enddate = len(self.storeweeklysales.loc[train_startdate:]) - 9
            train = self.storeweeklysales.loc[train_startdate:][:train_enddate]
            # test = self.storeweeklysales.loc[train_startdate:][train_enddate:-1]
            test = self.storeweeklysales[-9:-1]

            arima_model = auto_arima(self.storeweeklysales.loc[train_startdate:test.index[-1]], start_p=0, d=0,
                                     start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            forecast, CI = arima_model.predict(n_periods=8, return_conf_int=True, alpha=0.05)
            return pd.DataFrame([forecast, CI[:, 0], CI[:, 1], self.storeweeklysales[-9:-1].values],
                                index=['Forecast', 'lower CI', 'upper CI', 'Previous 8 Weeks']).T.set_index(
                forecastindex)


        else:
            storeweeklysales_dropzero = self.storeweeklysales[self.storeweeklysales != 0]
            train, test = (storeweeklysales_dropzero[:-9], storeweeklysales_dropzero[-9:-1])
            arima_model = auto_arima(storeweeklysales_dropzero[:test.index[-1]], start_p=0, d=0, start_q=0,
                                     max_p=5, max_d=5, max_q=5, start_P=0,
                                     D=1, start_Q=0, max_P=5, max_D=5,
                                     max_Q=5, m=2, seasonal=True,
                                     error_action='warn', trace=False,
                                     supress_warnings=True, stepwise=False,
                                     random=True,
                                     random_state=20, n_fits=50)

            forecast, CI = arima_model.predict(n_periods=8, return_conf_int=True, alpha=0.05)
            forecastresult1 = pd.DataFrame([forecast, CI[:, 0], CI[:, 1]],
                                           index=['Forecast', 'lower CI', 'upper CI']).T.set_index(forecastindex)

            forecastresult1['Previous 8 Weeks'] = self.storeweeklysales[-9:-1].values
            return forecastresult1

# # https://www.bootstrapcdn.com/bootswatch/

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("ROSSMANN Sales Dashboard",
                        className='text-center text-primary mb-4'),
                width=12)
    ),

    dbc.Row([

        dbc.Col([
            html.H5("Store Daily Sales:",
                   style={"textDecoration": "underline"}),

    html.Div(
        children=[
            html.Div(
            children=[html.Div(children='Store ID', className='menu-title'),
            dcc.Dropdown(id='my-dpdn', multi=True, value=[1, 2],
                         options=[{'label':x, 'value':x}
                                  for x in sorted(df['Store'].unique())], style={'width': '100%'}
                         )],),
            html.Div(
                children=[html.Div(children='Date', className='menu-title'),
                    dcc.DatePickerRange(id="date-range",
                    min_date_allowed=df.Date.min().date(),
                    max_date_allowed=datetime.date(2016, 9, 19),
                    start_date=df.Date.min().date(),
                    end_date=df.Date.max().date(), style={'width': '100%'})]
                )], className='menu'),

            dcc.Graph(id='line-fig', figure={})
        ], width={'size': 6},
           #xs=12, sm=12, md=12, lg=5, xl=5
        ),

        dbc.Col([
            html.H5("Store Weekly Sales:",
                   style={"textDecoration": "underline"}),


    html.Div(
        children=[
            html.Div(
            children=[html.Div(children='Store ID', className='menu-title'),
            dcc.Dropdown(id='my-dpdn2', multi=True, value=[1, 2],
                     options=[{'label':x, 'value':x}
                              for x in sorted(df['Store'].unique())], style={'width': '100%'}
                         )],),
        html.Div(
            children=[html.Div(children='Date', className='menu-title'),
                dcc.DatePickerRange(id="date-range2",
                min_date_allowed=df.Date.min().date(),
                max_date_allowed=datetime.date(2016, 9, 19),
                start_date=df.Date.min().date(),
                end_date=df.Date.max().date(), style={'width': '100%'})]
            )], className='menu'),

        dcc.Graph(id='line-fig2', figure={})
        ], width={'size': 6},
       #xs=12, sm=12, md=12, lg=5, xl=5
        ),


    ], no_gutters=False, justify='center'),  # Horizontal:start,center,end,between,around

    dbc.Row([

        dbc.Col([
            html.H5("Store Daily Customer:",
                   style={"textDecoration": "underline"}),

            html.Div(
                children=[
                    html.Div(
                        children=[html.Div(children='Store ID', className='menu-title'),
                                  dcc.Dropdown(id='my-dpdn3', multi=True, value=[1, 2],
                                               options=[{'label': x, 'value': x}
                                                        for x in sorted(df['Store'].unique())], style={'width': '100%'}
                                               )], ),
                    html.Div(
                        children=[html.Div(children='Date', className='menu-title'),
                                  dcc.DatePickerRange(id="date-range3",
                                                      min_date_allowed=df.Date.min().date(),
                                                      max_date_allowed=datetime.date(2016, 9, 19),
                                                      start_date=df.Date.min().date(),
                                                      end_date=df.Date.max().date(), style={'width': '100%'})]
                    )], className='menu'),

            dcc.Graph(id='line-fig3', figure={})
        ], width={'size': 6},
            # xs=12, sm=12, md=12, lg=5, xl=5
        ),



        dbc.Col([
            html.H5("Store Daily Spend Per Customer",
                   style={"textDecoration": "underline"}),

            html.Div(
                children=[
                    html.Div(
                        children=[html.Div(children='Store ID', className='menu-title'),
                                  dcc.Dropdown(id='my-dpdn4', multi=True, value=[1, 2],
                                               options=[{'label': x, 'value': x}
                                                        for x in sorted(df['Store'].unique())], style={'width': '100%'}
                                               )], ),
                    html.Div(
                        children=[html.Div(children='Date', className='menu-title'),
                                  dcc.DatePickerRange(id="date-range4",
                                                      min_date_allowed=df.Date.min().date(),
                                                      max_date_allowed=datetime.date(2016, 9, 19),
                                                      start_date=df.Date.min().date(),
                                                      end_date=df.Date.max().date(), style={'width': '100%'})]
                    )], className='menu'),

            dcc.Graph(id='line-fig4', figure={}),
        ], width={'size': 6},
           #xs=12, sm=12, md=12, lg=5, xl=5
        ),


    ], no_gutters=False, justify='center'),  # Vertical: start, center, end

     dbc.Row([

        dbc.Col([
            html.H5("Top 10 AverageDailySales by Store and Assortment Type:",
                   style={"textDecoration": "underline"}),
            # dcc.Dropdown(id='my-dpdn5', multi=False, value='',
            #               options=[{'label':x, 'value':x}
            #                        for x in sorted(sdf['StoreType'].unique())],
            #              ),

            dcc.Dropdown(id='my-dpdn5', multi=False, value='StoreType',
                          options=[{'label':x, 'value':x}
                                   for x in ['StoreType', 'Assortment']],
                         ),

            dcc.Dropdown(id='my-dpdn8', multi=False, value='a',
                          options=[{'label':x, 'value':x}
                                   for x in sorted(sdf['StoreType'].unique())],
                         ),

            dcc.Graph(id='bar-fig1', figure={}),
        ], width={'size': 6},
           #xs=12, sm=12, md=12, lg=5, xl=5
        ),


        dbc.Col([
            html.H5("Top 10 SpendPerCustomer by Store and Assortment Type:",
                   style={"textDecoration": "underline"}),

            dcc.Dropdown(id='my-dpdn6', multi=False, value='StoreType',
                          options=[{'label':x, 'value':x}
                                   for x in ['StoreType', 'Assortment']],
                         ),

            dcc.Dropdown(id='my-dpdn9', multi=False, value='a',
                          options=[{'label':x, 'value':x}
                                   for x in sorted(sdf['StoreType'].unique())],
                         ),
            dcc.Graph(id='bar-fig2', figure={}),
        ], width={'size': 6},
           #xs=12, sm=12, md=12, lg=5, xl=5
        )
     ], no_gutters=False, justify='center'),

    dbc.Row([

        dbc.Col([
            html.H5("Future 8 Weeks Forecasting:",
                   style={"textDecoration": "underline"}),
            dcc.Dropdown(id='my-dpdn7', multi=False, value=1,
                          options=[{'label':x, 'value':x}
                                   for x in sorted(df['Store'].unique())],
                         ),
            dcc.Graph(id='line-fig5', figure={}),
        ], width={'size': 6},
           #xs=12, sm=12, md=12, lg=5, xl=5
        )], no_gutters=False, justify='center'),


], fluid=True)


# Callback section: connecting the components
# ************************************************************************
# Line chart - multi daily sales

# @app.callback(
#     Output('line-fig', 'figure'),
#     Input('my-dpdn', 'value')
# )
# def update_graph(storeid):
#     dff = df[df['Store'].isin(storeid)].set_index('Date')
#     figln = px.line(dff, x=dff.index, y='Sales', color='Store')
#     return figln


@app.callback(
    Output('line-fig', 'figure'),
    Input('my-dpdn', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
)
def update_graph(storeid, start_date, end_date):
    mask=(df.Store.isin(storeid) & (df.Date >= start_date) & (df.Date <= end_date))
    dff = df.loc[mask,:].set_index('Date')
    figln = px.line(dff, x=dff.index, y='Sales', color='Store')
    return figln



# Line chart - multi weekly sales sum
@app.callback(
    Output('line-fig2', 'figure'),
    Input('my-dpdn2', 'value'),
    Input('date-range2', 'start_date'),
    Input('date-range2', 'end_date'),
)
def update_graph(storeid, start_date, end_date):
    mask = (df.Store.isin(storeid) & (df.Date >= start_date) & (df.Date <= end_date))
    dff = df.loc[mask,:].set_index('Date').groupby('Store')['Sales'].resample('w').sum().reset_index().set_index('Date')
    figln2 = px.line(dff, x=dff.index, y='Sales', color='Store', labels={'Sales':'Weekly Sales'})

    return figln2



# line chart - customer
@app.callback(
    Output('line-fig3', 'figure'),
    Input('my-dpdn3', 'value'),
    Input('date-range3', 'start_date'),
    Input('date-range3', 'end_date'),
)
def update_graph(storeid, start_date, end_date):
    mask = (df.Store.isin(storeid) & (df.Date >= start_date) & (df.Date <= end_date))
    dff = df.loc[mask,:].set_index('Date')
    figln3 = px.line(dff, x=dff.index, y='Customers', color='Store')
    return figln3



# line chart - sales per customer
@app.callback(
    Output('line-fig4', 'figure'),
    Input('my-dpdn4', 'value'),
    Input('date-range4', 'start_date'),
    Input('date-range4', 'end_date'),
)
def update_graph(storeid, start_date, end_date):
    mask = (df.Store.isin(storeid) & (df.Date >= start_date) & (df.Date <= end_date))
    dff = df.loc[mask,:].set_index('Date')
    dff['SalesPerCustomer'] = dff['Sales']/dff['Customers']
    figln4 = px.line(dff, x=dff.index, y='SalesPerCustomer', color='Store')
    return figln4


@app.callback(
    Output('bar-fig1', 'figure'),
    Input('my-dpdn5', 'value'),
    Input('my-dpdn8', 'value'),
)
# def update_graph(storetype):
#     sdff = sdf[sdf['StoreType']==storetype].sort_values('AverageDailySales', ascending=False).nlargest(10, 'AverageDailySales')
#     #figln5 = go.Figure(go.Bar(x=sdff['Sales'], y=sdff['Store'].values.astype('str'),  orientation='h'))
#     figbar1 = px.bar(x=sdff['AverageDailySales'], y=sdff['Store'].values.astype('str'),
#                      color=sdff['Store'],
#                      labels={'y':'Store ID', 'x':'Average Daily Sales'},  orientation='h')

def update_graph(typename, typeoption):
    sdff = sdf[sdf[typename] == typeoption].sort_values('AverageDailySales', ascending=False).nlargest(10, 'AverageDailySales')
    #figln5 = go.Figure(go.Bar(x=sdff['Sales'], y=sdff['Store'].values.astype('str'),  orientation='h'))
    figbar1 = px.bar(x=sdff['AverageDailySales'], y=sdff['Store'].values.astype('str'),
                     color=sdff['Store'],
                     labels={'y':'Store ID', 'x':'Average Daily Sales'},  orientation='h')
    return figbar1



@app.callback(
    Output('bar-fig2', 'figure'),
    Input('my-dpdn6', 'value'),
    Input('my-dpdn9', 'value'),
)
def update_graph(typename, typeoption):
    sdff = sdf[sdf[typename] == typeoption].sort_values('SalesPerCustomer', ascending=False).nlargest(10, 'SalesPerCustomer')
    #figln5 = go.Figure(go.Bar(x=sdff['Sales'], y=sdff['Store'].values.astype('str'),  orientation='h'))
    figbar2 = px.bar(x=sdff['SalesPerCustomer'], y=sdff['Store'].values.astype('str'),
                     color=sdff['Store'],
                     labels={'y':'Store ID', 'x':'Spend Per Customer'},  orientation='h')
    return figbar2



@app.callback(
    Output('line-fig5', 'figure'),
    Input('my-dpdn7', 'value'),
    # Input('date-range', 'start_date'),
    # Input('date-range', 'end_date'),
)
def update_graph(storeid):
    storeweeklysales = df[df.Store == storeid].set_index('Date').resample('w').Sales.sum()
    previous8w_index = storeweeklysales[-9:-1].index
    webresult = modeling(df, storeid).forecast('2015-07-27', '2015-09-20')
    fig5 = go.Figure([
        go.Scatter(
            name='Previous 8 Weeks',
            x=previous8w_index,
            y=webresult['Previous 8 Weeks'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            hoverinfo='none',
            name='',
            x=[previous8w_index[-1], webresult.index[0]],
            y=[storeweeklysales[-9:-1].values[-1], webresult['Forecast'][0]],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ),
        go.Scatter(
            name='Forcast',
            x=webresult.index,
            y=webresult['Forecast'],
            mode='lines',
            line=dict(color='rgb(255,0,0)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=webresult.index,
            y=webresult['upper CI'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=webresult.index,
            y=webresult['lower CI'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig5.update_layout(
        width=900,
        height=500,
        title=f'Store {storeid}',
        yaxis_title='Weekly Sales',
        hovermode="x"
    )

    return fig5




# Histogram
# @app.callback(
#     Output('my-hist', 'figure'),
#     Input('my-checklist', 'value')
# )
# def update_graph(storeid):
#     dff = df[df['Store'].isin(storeid)]
#     #dff = dff[dff['Store']==storeid]
#     fighist = px.histogram(dff, x='Sales', nbins=10)
#     return fighist


if __name__=='__main__':
    app.run_server(debug=True, port=8000)


