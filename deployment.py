import joblib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

model = joblib.load('RandomForest.pkl')
encoders = joblib.load('label_encoder.pkl')  
scaler = joblib.load('scaler')

st.set_page_config(page_title='Sales Forecasting and Optimization', page_icon='logo.png', initial_sidebar_state='expanded')


st.sidebar.markdown("### Menu")
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "Plotly Dashboard", "Power BI Dashboard"])

if page == "Home":
    st.image('image.png')
    st.markdown("<h1 >Dataset Overview</h1>", unsafe_allow_html=True)

    st.markdown("""
    ## **Data Description**

    This project uses historical retail sales data sourced from 
    [Kaggle - Sales Forecasting Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting).
    
    The dataset (`train.csv`) contains **9,800 rows** and **18 columns** describing customer details, 
    order information, product categories, and financial transactions.
    
    The goal of this project is to build a machine learning model that forecasts the **Sales** amount—our target variable—based on available features. This will support strategic decisions in **sales optimization**, **inventory planning**, and **customer segmentation**.
    """)

    # Data Dictionary Table
    sales_data_dict = pd.DataFrame({
        'Column': [
            'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID',
            'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code',
            'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Sales'
        ],
        'Datatype': [
            'Number', 'Text', 'Date', 'Date', 'Text', 'Text',
            'Text', 'Text', 'Text', 'Text', 'Text', 'Number',
            'Text', 'Text', 'Text', 'Text', 'Text', 'Number'
        ],
        'Description': [
            'Unique numeric identifier for each row in the dataset.',
            'Unique identifier for each order.',
            'The date when the order was placed (Format: DD/MM/YYYY).',
            'The date when the order was shipped (Format: DD/MM/YYYY).',
            'Shipping method used (e.g., Second Class, Standard Class).',
            'Unique identifier for each customer.',
            'Name of the customer who placed the order.',
            'Customer segment (e.g., Consumer, Corporate, Home Office).',
            'Country of the customer.',
            'City of the customer.',
            'State of the customer.',
            'Postal/ZIP code (may contain missing values).',
            'Geographic region of the customer (e.g., West, South).',
            'Unique identifier for each product.',
            'Main category of the product (e.g., Furniture, Office Supplies, Technology).',
            'Sub-category of the product (e.g., Chairs, Labels).',
            'Name of the product.',
            '**Target** — Sale amount (in USD) for the product line.'
        ]
        
    })

    st.markdown("## Data Dictionary")
    st.dataframe(sales_data_dict, use_container_width=True)

  
    st.markdown(f"""
        <style>
            .hover-div-presentation {{
                padding: 10px;
                border-radius: 10px;
                background-color: #211C84;
                margin-bottom: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: background-color 0.3s ease, box-shadow 0.3s ease;
                cursor: pointer;
                text-decoration: none;
            }}
            .hover-div-presentation:hover {{
                background-color: #4D55CC;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
            }}
            .hover-div-notebook {{
                padding: 10px;
                border-radius: 10px;
                background-color: #7A73D1;
                margin-bottom: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: background-color 0.3s ease, box-shadow 0.3s ease;
                cursor: pointer;
                text-decoration: none;
            }}
            .hover-div-notebook:hover {{
                background-color: #B5A8D5;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
            }}
            h4 {{
                margin: 0;
                text-align: center;
            }}
        </style>

        <!-- Presentation Link -->
        <a href="https://drive.google.com/file/d/1nbsqG7YZlC3KFYzQKmg6yP_JfLzNFmrN/view?usp=sharing" target="_blank" class="hover-div-presentation">
            <h4 style="color: white;"> View Our Presentation</h4>
        </a>

        <!-- Notebook Link -->
        <a href="https://github.com/TokaKhaled4/Sales_Forecasting_and_Optimization_Project/blob/main/Sales%20Forcasting%20and%20Optimiztion.ipynb" target="_blank" class="hover-div-notebook">
            <h4 style="color: white;">View Our Notebook</h4>
        </a>
    """, unsafe_allow_html=True)

elif page == "Prediction":
    st.markdown("<h1 style='text-align: center;'>Sales Forecasting and Optimization</h1>", unsafe_allow_html=True)

    st.subheader("Enter Order Details")

    Sub_Category = st.selectbox("Sub-Category", encoders['Sub_Category'].classes_)
    Region = st.selectbox("Region", encoders['Region'].classes_)
    State = st.text_input("State")
    City = st.text_input("City")
    Order_Date = st.date_input("Order Date")
    Ship_Date = st.date_input("Ship Date")
    Postal_Code = st.number_input("Postal Code", step=1)
    Ship_Mode = st.selectbox("Ship Mode", encoders['Ship_Mode'].classes_)
    Product_Name = st.text_input("Product Name (e.g., 'Fellowes SuperStor')").lower()
    Category = st.selectbox("Category", encoders['Category'].classes_)

    if st.button("Predict Sales"):

        try:
            # Derived features
            Shipping_Duration = (Ship_Date - Order_Date).days
            Month = Order_Date.month

            # Clean product name
            import re
            Product_Name_clean = re.sub(r'[^\w\s]', '', Product_Name).strip()

            # Construct input dictionary
            input_dict = {
                'Sub_Category': Sub_Category,
                'Region': Region,
                'State': State,
                'City': City,
                'Shipping_Duration': Shipping_Duration,
                'Month': Month,
                'Postal_Code': Postal_Code,
                'Ship_Mode': Ship_Mode,
                'Product_Name': Product_Name_clean,
                'Category': Category
            }

            # Encode categorical columns
            for col in ['Sub_Category', 'Region', 'State', 'City', 'Ship_Mode', 'Product_Name', 'Category']:
                if input_dict[col] not in encoders[col].classes_:
                    st.error(f"Unknown value '{input_dict[col]}' in column '{col}' — please choose a known one.")
                    st.stop()
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]

            # Final feature array
            features = np.array([list(input_dict.values())])

            # Scale
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)[0]
            st.success(f"📦 Predicted Sales: **{prediction}**")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


elif page == "Plotly Dashboard":
    st.markdown("<h1 style='text-align: center;'>Plotly Sales Dashboard</h1>", unsafe_allow_html=True)

    # Load dataset
    df = pd.read_csv('train.csv', parse_dates=['Order Date', 'Ship Date'])
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

    # RFM Analysis
    from datetime import timedelta
    analysis_date = df['Order Date'].max() + timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (analysis_date - x.max()).days,
        'Order ID': 'nunique',
        'Sales': 'sum'
    }).reset_index()
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)

    # Enrich df
    df = df.merge(rfm, on='Customer ID', how='left')
    df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Delivery Time (days)'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Colors
    custom_palette = ['#211C84', '#4D55CC', '#7A73D1', '#B5A8D5']
    churn_color_map = {0: '#4D55CC', 1: '#B5A8D5'}

    # Plotly charts
    import plotly.express as px

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
        px.bar(df.groupby('Churn')['Sales'].sum().reset_index(), x='Churn', y='Sales', color='Churn', title='Sales: Churned vs Active', color_discrete_map=churn_color_map),
        px.bar(rfm.groupby('Churn')['Frequency'].mean().reset_index(), x='Churn', y='Frequency', color='Churn', title='Avg Frequency: Churned vs Active', color_discrete_map=churn_color_map),
        px.bar(df.groupby(['Category', 'Churn'])['Sales'].sum().reset_index(), x='Category', y='Sales', color='Churn', barmode='group', title='Sales by Category & Churn', color_discrete_map=churn_color_map),
        px.histogram(rfm, x='Recency', nbins=30, color='Churn', title='Recency by Churn', color_discrete_map=churn_color_map),
        px.treemap(rfm, path=['Customer ID'], values='Monetary', color='Churn', title='Customer Value Treemap', color_discrete_map=churn_color_map),
    ]

    # Display each figure in Streamlit
    for fig in figs:
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


elif page == "Power BI Dashboard":
    st.markdown("<h1 style='text-align: center;'>Power BI Sales Dashboard</h1>", unsafe_allow_html=True)

    powerbi_html = """
    <iframe title="Sales Forecasting and Optimization" width="100%" height="600"
    src="https://app.powerbi.com/view?r=eyJrIjoiNTJiNGE0NjYtMDljMy00ZGNhLWI0NmMtMjYxOGQ3NmY5OGJjIiwidCI6IjIwODJkZTQ2LTFhZmEtNGI2NC1hNDQwLTY1NThmODBlOTg0MCIsImMiOjh9"
    frameborder="0" allowFullScreen="true"></iframe>
    """
    
    st.markdown(powerbi_html, unsafe_allow_html=True)
