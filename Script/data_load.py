import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress
import seaborn as sns

# 1. Load and Clean Data
# ===================
def load_and_clean_data(data_path):
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded: {data_path} | Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        df.drop_duplicates(inplace=True)
        df.dropna(how='all', inplace=True)
        df = df.convert_dtypes()
        print(f"Cleaned: Removed duplicates and empty rows.")
        return df
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return pd.DataFrame()

    
# 2.  Handle Missing Values
def handle_missing_values(df, strategy='drop'):
    try:
        print("Missing values before handling:\n", df.isnull().sum()[df.isnull().sum() > 0])
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        print("Missing values handled.")
        return df
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return df

# 3.  Exploratory Data Analysis (EDA)
# ===================================
def eda_univariate(df, columns):
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution: {col}")
            plt.show()

def eda_bivariate(df, target):
    for col in df.select_dtypes(include=np.number).columns:
        if col != target:
            plt.figure()
            sns.boxplot(x=target, y=col, data=df)
            plt.title(f"{col} vs {target}")
            plt.show()


# 4. IP Conversion & Merge
# ========================
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return np.nan

def merge_ip_geolocation(fraud_df, ip_df):
    try:
        fraud_df['ip_int'] = (fraud_df['ip_address'].apply(ip_to_int).astype(int))
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)

        merged_df = pd.merge_asof(
            fraud_df.sort_values('ip_int'),
            ip_df.sort_values('lower_bound_ip_address'),
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        print("IP address merged with geolocation successfully.")
        return merged_df
    except Exception as e:
        print(f"Error merging IP geolocation: {e}")
        return fraud_df
