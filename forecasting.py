import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

Project 1: Sales Analysis and Forecasting

Load dataset (replace with your dataset path or download a sales dataset)
dataset_url = "https://raw.githubusercontent.com/selva86/datasets/master/RetailSales.csv"
sales_data = pd.read_csv(dataset_url, parse_dates=['date'], index_col='date')

Display basic info about the dataset
print("Dataset Head:")
print(sales_data.head())
print("\nDataset Info:")
print(sales_data.info())

Check for missing values
if sales_data.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    sales_data = sales_data.fillna(method='ffill')

Exploratory Data Analysis
plt.figure(figsize=(12, 6))
plt.plot(sales_data.index, sales_data['value'], label='Sales Over Time')
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

Train-test split for time series
train_size = int(len(sales_data) * 0.8)
train, test = sales_data[:train_size], sales_data[train_size:]

Fit ARIMA model
print("\nFitting ARIMA model...")
model = ARIMA(train['value'], order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

Forecast
print("\nForecasting...")
forecast = model_fit.forecast(steps=len(test))

Plot results
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

Evaluate model
mae = np.mean(np.abs(forecast - test['value']))
print(f"Mean Absolute Error (MAE): {mae}")

Project 2: Customer Segmentation Using Clustering

Load dataset (replace with your dataset path or use a public dataset)
cust_dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
customer_data = pd.read_csv(cust_dataset_url)

Display basic info about the dataset
print("\nCustomer Dataset Head:")
print(customer_data.head())
print("\nCustomer Dataset Info:")
print(customer_data.info())

Selecting features for clustering
features = customer_data.iloc[:, :-1]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
customer_data['Cluster'] = kmeans.labels_

Visualizing clusters (only first two features for simplicity)
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

Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

Project 3: Bank Data Comparison

Load CSV and SAP data (replace with your file paths)
csv_data = pd.read_csv("bank_csv_data.csv")
sap_data = pd.read_csv("bank_sap_data.csv")

Display basic info about both datasets
print("\nCSV Data Head:")
print(csv_data.head())
print("\nSAP Data Head:")
print(sap_data.head())

Merge data for comparison
merged_data = pd.merge(csv_data, sap_data, on='Transaction_ID', how='outer', indicator=True)

Identify mismatches
mismatches = merged_data[merged_data['_merge'] != 'both']

Save mismatches to a new CSV
output_file = "bank_data_mismatches.csv"
mismatches.to_csv(output_file, index=False)
print(f"\nMismatches saved to {output_file}")

Display summary of mismatches
print("\nSummary of Mismatches:")
print(mismatches['_merge'].value_counts())
