import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score

# Function to load data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(df):
    df = df.fillna(0)
    X = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']
    return X, y

# Function to perform feature selection
def select_features(X, y):
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

# Function to train models and display results
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "LightGBM": lgb.LGBMClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'model': model}
    return results

# Streamlit app
st.title('Bankruptcy Prediction Model')
st.write('This application allows you to load a dataset, preprocess it, and train various machine learning models for bankruptcy prediction.')

# File uploader
uploaded_file = st.file_uploader("Bankruptcy Prediction.csv", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write(df.head())

    # Data preprocessing
    X, y = preprocess_data(df)
    st.write('Data Preprocessed Successfully!')

    # Feature selection
    X_selected, selected_features = select_features(X, y)
    st.write('Selected Features:', selected_features)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    st.write('Class Imbalance Handled Using SMOTE!')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Display results
    st.write('Model Training Results:')
    for name, result in results.items():
        st.write(f"{name} - Accuracy: {result['accuracy']:.2f}, ROC AUC: {result['roc_auc']:.2f}")

    # Visualize ROC curves
    st.write('ROC Curves:')
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['model'].predict_proba(X_test)[:, 1] if hasattr(result['model'], "predict_proba") else result['model'].decision_function(X_test))
        plt.plot(fpr, tpr, label=f'{name} (area = {result["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    st.pyplot(plt)
