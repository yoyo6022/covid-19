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
sys.path.insert(1, '/Users/liuyang/Desktop/Springboard_Capstone3/notebook')
from modelingclass import modeling

df = pd.read_csv("~/Desktop/Springboard_Capstone3/data/train.csv")
df.Date=pd.to_datetime(df.Date, format='%Y-%m-%d')
sdf = pd.read_csv("~/Desktop/Springboard_Capstone3/data/combined_data.csv")



# # https://www.bootstrapcdn.com/bootswatch/

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("Rossmann Sales Dashboard",
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
            dcc.Dropdown(id='my-dpdn', multi=True, value='',
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
            dcc.Dropdown(id='my-dpdn2', multi=True, value='',
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
                                  dcc.Dropdown(id='my-dpdn3', multi=True, value='',
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
                                  dcc.Dropdown(id='my-dpdn4', multi=True, value='',
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
            html.H5("Top 10 AverageDailySales by StoreType:",
                   style={"textDecoration": "underline"}),
            dcc.Dropdown(id='my-dpdn5', multi=False, value='',
                          options=[{'label':x, 'value':x}
                                   for x in sorted(sdf['StoreType'].unique())],
                         ),
            dcc.Graph(id='bar-fig1', figure={}),
        ], width={'size': 6},
           #xs=12, sm=12, md=12, lg=5, xl=5
        ),


        dbc.Col([
            html.H5("Top 10 SpendPerCustomer by StoreType:",
                   style={"textDecoration": "underline"}),
            dcc.Dropdown(id='my-dpdn6', multi=False, value='',
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
            dcc.Dropdown(id='my-dpdn7', multi=False, value='',
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
    Input('my-dpdn5', 'value')
)
def update_graph(storetype):
    sdff = sdf[sdf['StoreType']==storetype].sort_values('AverageDailySales', ascending=False).nlargest(10, 'AverageDailySales')
    #figln5 = go.Figure(go.Bar(x=sdff['Sales'], y=sdff['Store'].values.astype('str'),  orientation='h'))
    figbar1 = px.bar(x=sdff['AverageDailySales'], y=sdff['Store'].values.astype('str'),
                     color=sdff['Store'],
                     labels={'y':'Store ID', 'x':'Average Daily Sales'},  orientation='h')
    return figbar1



@app.callback(
    Output('bar-fig2', 'figure'),
    Input('my-dpdn6', 'value')
)
def update_graph(storetype):
    sdff = sdf[sdf['StoreType']==storetype].sort_values('SalesPerCustomer', ascending=False).nlargest(10, 'SalesPerCustomer')
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
    webresult = modeling(df, storeid).forecast('2015-07-27', '2015-09-20')
    fig5 = go.Figure([
        go.Scatter(
            name='Forcast',
            x=webresult.index,
            y=webresult['Forecast'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
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
        width=600,
        height=400,
        yaxis_title='Weekly Sales ',
        title='Continuous, variable value error bars',
        hovermode="x"
    )
    #fig5.show()

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


