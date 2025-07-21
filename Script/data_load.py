import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



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
        fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int).astype(int)

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
    

# 5. Feature Engineering
def feature_engineering_fraud(df):
    try:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

        df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        df['avg_time_between'] = df.sort_values('purchase_time').groupby('user_id')['purchase_time'].diff().dt.total_seconds()

        print(" Feature engineering completed.")
        return df
    except Exception as e:
        print(f" Error in feature engineering: {e}")
        return df

#  6. Data Transformation
# a) Class Imbalance
def balance_classes(X, y, method='smote'):
    try:
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        else:
            sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"Class imbalance handled using {method}.")
        return X_res, y_res
    except Exception as e:
        print(f"Error balancing classes: {e}")
        return X, y

# b) Scaling & Encoding
def scale_data(X, method='standard'):
    try:
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        print(f"Data scaled using {method}.")
        return X_scaled
    except Exception as e:
        print(f"Error scaling data: {e}")
        return X

def encode_categorical(df, cat_cols):
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(cat_cols, axis=1).reset_index(drop=True)
        df = pd.concat([df.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
        print(" Categorical variables encoded.")
        return df
    except Exception as e:
        print(f" Error encoding categorical columns: {e}")
        return df
