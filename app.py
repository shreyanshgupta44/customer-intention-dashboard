import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess
df = pd.read_csv("online_shoppers_intention.csv")

# Encode categorical
for col in ['Month', 'VisitorType', 'Weekend']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['Revenue'])
y = df['Revenue']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled =
