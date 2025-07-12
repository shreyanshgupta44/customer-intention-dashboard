import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Title
st.title("🧠 Customer Purchase Intention Dashboard")
st.markdown("Analyze and visualize purchase behavior predictions using ML models.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")

df = load_data()
st.write("### Sample Data", df.head())

# Encode categorical features
categorical_cols = ['Month', 'VisitorType', 'Weekend']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Revenue'])
y = df['Revenue']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000)
stack = StackingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], final_estimator=lr)

# Train
rf.fit(X_train_scaled, y_train)
stack.fit(X_train_scaled, y_train)

# Predict
rf_pred = rf.predict(X_test_scaled)
stack_pred = stack.predict(X_test_scaled)

# Metrics
metrics = {
    "Accuracy": [accuracy_score(y_test, rf_pred), accuracy_score(y_test, stack_pred)],
    "Precision": [precision_score(y_test, rf_pred), precision_score(y_test, stack_pred)],
    "Recall": [recall_score(y_test, rf_pred), recall_score(y_test, stack_pred)],
    "F1 Score": [f1_score(y_test, rf_pred), f1_score(y_test, stack_pred)]
}
model_names = ["Random Forest", "Stacked Model"]

# Plot
for metric, values in metrics.items():
    st.subheader(f"{metric} Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x=model_names, y=values, palette="viridis", ax=ax)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Feature Importances
importances = rf.feature_importances_
st.subheader("📊 Feature Importances (Random Forest)")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=X.columns, ax=ax2)
st.pyplot(fig2)
