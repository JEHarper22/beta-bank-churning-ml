import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load dataset from a given file path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the DataFrame by removing duplicates and handling incorrect data."""
    df = df.drop_duplicates()
    # Further cleaning steps can be added here
    return df

def encode_categorical_features(df, categorical_cols):
    """Encode categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    return pd.concat([df.drop(categorical_cols, axis=1), pd.DataFrame(encoded_cols)], axis=1)

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    return df.fillna(df.mean())  # Example: fill with mean for numerical columns

def feature_scaling(df, scale_cols):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    return df

def split_data(df, target_column):
    """Split the data into train, validation, and test sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test