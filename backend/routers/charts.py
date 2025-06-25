import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features
from sklearn.preprocessing import LabelEncoder
import os

# Load and prepare data
data = load_all_data()
df = data["train"]
df = prepare_features(df, data["oil"], data["holidays"], data["transactions"], data["stores"])
df.dropna(inplace=True)

# Encode necessary categorical columns
le = LabelEncoder()
for col in ['store_nbr', 'family', 'city', 'state', 'type']:
    df[col] = le.fit_transform(df[col])

# Create time-based columns
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')

# ----------- DYNAMIC CHART FUNCTIONS  -----------

def plot_sales_by_month():
    monthly_sales = df.groupby('month')['sales'].sum().reset_index()
    fig = px.line(monthly_sales, x='month', y='sales', title='Monthly Sales Trend')
    return fig

def plot_sales_by_store():
    sales_by_store = df.groupby('store_nbr')['sales'].sum().reset_index()
    fig = px.bar(sales_by_store, x='store_nbr', y='sales', title='Total Sales by Store')
    return fig

def plot_sales_by_family():
    sales_by_family = df.groupby('family')['sales'].sum().reset_index()
    fig = px.pie(sales_by_family, names='family', values='sales', title='Sales Distribution by Product Family')
    return fig

def plot_dcoilwtico_vs_sales():
    fig = px.scatter(df, x='dcoilwtico', y='sales', title='Oil Price vs Sales', trendline='ols')
    return fig

# Dictionary for dynamic chart access
plots = {
    "Monthly Sales": plot_sales_by_month,
    "Sales by Store": plot_sales_by_store,
    "Sales by Family": plot_sales_by_family,
    "Oil Price vs Sales": plot_dcoilwtico_vs_sales
}

# ----------- STATIC CHART GENERATION -----------

def save_all_static_charts(output_dir="static/charts"):
    os.makedirs(output_dir, exist_ok=True)
    for name, plot_func in plots.items():
        fig = plot_func()
        file_name = name.lower().replace(" ", "_") + ".png"
        fig.write_image(os.path.join(output_dir, file_name))
    print("✅ Static charts saved to:", output_dir)


save_all_static_charts()
