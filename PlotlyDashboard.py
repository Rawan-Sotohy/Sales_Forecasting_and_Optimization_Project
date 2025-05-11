import pandas as pd
import plotly.express as px
from datetime import timedelta
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

df = pd.read_csv('train.csv', parse_dates=['Order Date', 'Ship Date'])

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

analysis_date = df['Order Date'].max() + timedelta(days=1)

# Calculate RFM 
rfm = df.groupby('Customer ID').agg({
    'Order Date': lambda x: (analysis_date - x.max()).days,
    'Order ID': 'nunique',
    'Sales': 'sum'
}).reset_index()
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

rfm['Churn'] = (rfm['Recency'] > 90).astype(int)

df = df.merge(rfm, on='Customer ID', how='left')
df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
df['Delivery Time (days)'] = (df['Ship Date'] - df['Order Date']).dt.days

custom_palette = ['#211C84', '#4D55CC', '#7A73D1', '#B5A8D5']
churn_color_map = {0: '#4D55CC', 1: '#B5A8D5'}

figs = [
    px.histogram(rfm, x='Recency', nbins=30, title='Recency Distribution', color_discrete_sequence=[custom_palette[0]]),
    px.histogram(rfm, x='Frequency', nbins=20, title='Frequency Distribution', color_discrete_sequence=[custom_palette[1]]),
    px.histogram(rfm, x='Monetary', nbins=30, title='Monetary Value Distribution', color_discrete_sequence=[custom_palette[2]]),
    px.bar(rfm.sort_values('Monetary', ascending=False).head(20), x='Customer ID', y='Monetary', title='Top 20 Customers by Sales', color_discrete_sequence=[custom_palette[3]]),
    px.line(df.groupby('Month')['Sales'].sum().reset_index(), x='Month', y='Sales', title='Monthly Sales Trend', color_discrete_sequence=[custom_palette[0]]),
    px.bar(df.groupby('Segment')['Sales'].sum().reset_index(), x='Segment', y='Sales', title='Sales by Segment', color_discrete_sequence=[custom_palette[1]]),
    px.sunburst(df, path=['Category', 'Sub-Category'], values='Sales', title='Sales by Category & Sub-Category', color_discrete_sequence=custom_palette),
    px.box(df, x='Ship Mode', y='Delivery Time (days)', title='Delivery Time by Ship Mode', color_discrete_sequence=[custom_palette[2]]),
    px.pie(rfm, names='Churn', title='Churn Distribution', color='Churn', color_discrete_map=churn_color_map),
    px.bar(df.groupby('Churn')['Sales'].sum().reset_index(), x='Churn', y='Sales', color='Churn', title='Sales: Churned vs Active', color_continuous_scale=custom_palette[1:3]),
    px.bar(rfm.groupby('Churn')['Frequency'].mean().reset_index(),
    x='Churn', y='Frequency', color='Churn',
    title='Avg Frequency: Churned vs Active',
    color_continuous_scale=custom_palette[1:3]),
    px.bar(df.groupby(['Category', 'Churn'])['Sales'].sum().reset_index(), x='Category', y='Sales', color='Churn', barmode='group', title='Sales by Category & Churn', color_continuous_scale=custom_palette[1:3]),
    px.histogram(rfm, x='Recency', nbins=30, color='Churn', title='Recency by Churn', color_discrete_map=churn_color_map),
    px.treemap(rfm, path=['Customer ID'], values='Monetary', color='Churn', title='Customer Value Treemap', color_continuous_scale=custom_palette),
]

for fig in figs:
    fig.update_layout(height=700)

app.layout = dbc.Container([
    html.H1("Sales Forcasting Dashboards", className='text-center my-4'),

    *[
        html.Div([
            dcc.Graph(figure=fig),
            html.Hr()
        ]) for fig in figs
    ]

], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
