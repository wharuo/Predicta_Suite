import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_sales_data(url=None):
    """
    Load sales data from a CSV file or URL.
    """
    try:
        if url is None:
            url = "https://raw.githubusercontent.com/selva86/datasets/master/RetailSales.csv"
        sales_data = pd.read_csv(url, parse_dates=['date'], index_col='date')
        if sales_data.isnull().sum().sum() > 0:
            print("Filling missing values in sales data...")
            sales_data = sales_data.fillna(method='ffill')
        return sales_data
    except Exception as e:
        print(f"Error loading sales data: {e}")
        return None

def plot_sales(sales_data, show_plot=True):
    """
    Plot sales over time.
    """
    if 'value' not in sales_data.columns:
        print("Sales data must have a 'value' column.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data.index, sales_data['value'], label='Sales Over Time')
    plt.title("Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    if show_plot:
        plt.show()

def arima_forecasting(sales_data, order=(5, 1, 0), show_plot=True):
    """
    Forecast sales using ARIMA model.
    """
    if 'value' not in sales_data.columns:
        print("Sales data must have a 'value' column.")
        return
    train_size = int(len(sales_data) * 0.8)
    train, test = sales_data[:train_size], sales_data[train_size:]
    model = ARIMA(train['value'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    mae = np.mean(np.abs(forecast - test['value']))
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train['value'], label='Train')
        plt.plot(test.index, test['value'], label='Test')
        plt.plot(test.index, forecast, label='Forecast', linestyle='--')
        plt.title("Sales Forecasting")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid()
        plt.show()

def load_customer_data(url=None):
    """
    Load customer data for segmentation.
    """
    try:
        if url is None:
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        customer_data = pd.read_csv(url)
        return customer_data
    except Exception as e:
        print(f"Error loading customer data: {e}")
        return None

def customer_segmentation(customer_data, n_clusters=3, show_plot=True):
    """
    Segment customers using KMeans clustering.
    """
    features = customer_data.select_dtypes(include=[np.number])
    if features.empty:
        print("No numeric features found for clustering.")
        return
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    customer_data['Cluster'] = kmeans.labels_
    print("Customer segmentation completed.")
    if show_plot:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            hue=customer_data['Cluster'],
            palette='viridis',
            legend='full'
        )
        plt.title("Customer Clusters")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    print("\nCluster Centers (first two features):")
    print(kmeans.cluster_centers_[:, :2])

def compare_bank_data(csv_path="bank_csv_data.csv", sap_path="bank_sap_data.csv", output_file="bank_data_mismatches.csv"):
    """
    Compare two bank datasets and output mismatches.
    """
    try:
        csv_data = pd.read_csv(csv_path)
        sap_data = pd.read_csv(sap_path)
        if 'Transaction_ID' not in csv_data.columns or 'Transaction_ID' not in sap_data.columns:
            print("Both datasets must have a 'Transaction_ID' column.")
            return
        merged_data = pd.merge(csv_data, sap_data, on='Transaction_ID', how='outer', indicator=True)
        mismatches = merged_data[merged_data['_merge'] != 'both']
        mismatches.to_csv(output_file, index=False)
        print(f"\nMismatches saved to {output_file}")
        print("\nSummary of Mismatches:")
        print(mismatches['_merge'].value_counts())
    except Exception as e:
        print(f"Error comparing bank data: {e}")

if __name__ == "__main__":
    # Sales Forecasting
    sales_data = load_sales_data()
    if sales_data is not None:
        plot_sales(sales_data)
        arima_forecasting(sales_data)

    # Customer Segmentation
    customer_data = load_customer_data()
    if customer_data is not None:
        customer_segmentation(customer_data)

    # Bank Data Comparison
    compare_bank_data()
