import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# Load data function
@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv("Bankruptcy Prediction.csv")

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    df = df.fillna(0)
    
    # Encode categorical variables if any
    # Example: df = pd.get_dummies(df, columns=['categorical_column'])
    
    return df

# Function to train and evaluate models
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report

# Main function to run Streamlit app
def main():
    # Title of the app
    st.title('Bankruptcy Prediction App')

    # Load the dataset
    df = load_data()

    # Preprocess the data
    df = preprocess_data(df)

    # Sidebar inputs for user interaction
    st.sidebar.header('User Input Features')

    # Display a few rows of the dataset
    st.subheader('Dataset')
    st.write(df.head())

    # Split data into X and y
    X = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model selection
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

    # Train and evaluate the selected model
    st.subheader('Model Training and Evaluation')
    selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))
    if st.button('Train Model'):
        model = models[selected_model]
        trained_model, accuracy, report = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        st.write(f'Selected Model: {selected_model}')
        st.write(f'Model Accuracy: {accuracy:.2f}')
        st.write('Classification Report:')
        st.write(report)

        # Store trained model in session state
        st.session_state['trained_model'] = trained_model

    # Example prediction
    st.sidebar.header('Example Prediction')

    # User inputs for prediction
    user_inputs = {}
    for feature in X.columns:
        user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))

    # Predict function
    def predict(model, user_inputs):
        input_data = pd.DataFrame(user_inputs, index=[0])
        prediction = model.predict(input_data)
        return prediction

    # Make predictions on user input
    if st.sidebar.button('Predict'):
        if 'trained_model' not in st.session_state:
            st.warning('Please train a model first!')
        else:
            model = st.session_state['trained_model']
            prediction = predict(model, user_inputs)
            st.subheader('Prediction')
            if prediction[0] == 1:
                st.write('The company is predicted to go bankrupt.')
            else:
                st.write('The company is predicted not to go bankrupt.')

if __name__ == '__main__':
    main()
