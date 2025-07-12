import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("online_shoppers_intention.csv")
    
    # Encode categorical features
    df['Month'] = LabelEncoder().fit_transform(df['Month'])
    df['VisitorType'] = LabelEncoder().fit_transform(df['VisitorType'])
    df['Weekend'] = LabelEncoder().fit_transform(df['Weekend'])
    
    return df

df = load_data()

# Define features and target
X = df[['ProductRelated', 'Month']]  # only using two features for now
y = df['Revenue']

# Split and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🛒 Purchase Prediction")
st.markdown("Enter customer details to check if they will purchase or not.")

# User input
product_related = st.number_input("Number of Product Pages Viewed", min_value=0, max_value=500, value=10)

month_map = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
    'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}
month = st.selectbox("Month of Visit", list(month_map.keys()))
month_encoded = month_map[month]

# Predict
if st.button("Predict"):
    input_data = scaler.transform([[product_related, month_encoded]])
    prediction = model.predict(input_data)[0]
    
    if prediction:
        st.success("✅ This customer is **likely to make a purchase.**")
    else:
        st.error("❌ This customer is **unlikely to make a purchase.**")



