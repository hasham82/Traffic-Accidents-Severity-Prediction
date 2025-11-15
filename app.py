import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Traffic Accident Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Traffic Accident Severity Prediction System")
st.markdown("---")

# Sidebar for file upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Accidents CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ Data loaded successfully!")
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Show raw data
        with st.expander("📊 View Raw Data"):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Clean and format
        st.subheader("🔧 Data Preprocessing")
        with st.spinner("Processing data..."):
            df.columns = df.columns.str.strip()
            cat_cols = ['Location', 'Weather_Condition', 'Road_Condition', 'Light_Condition', 'Vehicle_Type']
            df[cat_cols] = df[cat_cols].astype(str).apply(lambda col: col.str.strip().str.lower())
            df['Severity'] = df['Severity'].str.strip().str.capitalize()
            
            # Drop invalid rows
            initial_count = len(df)
            df = df[(df['Speed_kmph'] > 0) & (df['Age_of_Driver'] > 0)]
            removed_count = initial_count - len(df)
            
            st.info(f"Removed {removed_count} invalid records (negative speed/age)")
            
            # Encode target
            le = LabelEncoder()
            df['Severity_Label'] = le.fit_transform(df['Severity'])
            
            # Features and target
            X = df.drop(['Accident_ID', 'Severity', 'Severity_Label'], axis=1)
            y = df['Severity_Label']
            
            # Preprocessing pipeline
            numeric_features = ['Speed_kmph', 'Age_of_Driver']
            categorical_features = ['Location', 'Weather_Condition', 'Road_Condition', 
                                   'Light_Condition', 'Vehicle_Type']
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
            # Model pipeline
            clf_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
            ])
        
        st.success("✅ Data preprocessing completed!")
        
        # Train model
        st.subheader("🤖 Model Training")
        with st.spinner("Training Random Forest model..."):
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            clf_pipeline.fit(X_train, y_train)
            
            # Calculate accuracy
            train_score = clf_pipeline.score(X_train, y_train)
            test_score = clf_pipeline.score(X_test, y_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_score:.2%}")
        with col2:
            st.metric("Testing Accuracy", f"{test_score:.2%}")
        
        # Feature importances
        st.subheader("📈 Feature Importance Analysis")
        
        model = clf_pipeline.named_steps['classifier']
        ohe = clf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        ohe_features = ohe.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_features)
        
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
        
        # Plot top 10
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax)
        ax.set_title('Top 10 Important Features for Accident Severity', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show importance table
        with st.expander("📋 View Feature Importance Table"):
            st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)
        
        # Prediction section
        st.markdown("---")
        st.subheader("🔮 Make Predictions")
        st.write("Enter accident details to predict severity:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            speed = st.number_input("Speed (km/h)", min_value=1, max_value=200, value=60)
            age = st.number_input("Driver Age", min_value=16, max_value=100, value=30)
            location = st.selectbox("Location", df['Location'].unique())
            weather = st.selectbox("Weather Condition", df['Weather_Condition'].unique())
        
        with col2:
            road = st.selectbox("Road Condition", df['Road_Condition'].unique())
            light = st.selectbox("Light Condition", df['Light_Condition'].unique())
            vehicle = st.selectbox("Vehicle Type", df['Vehicle_Type'].unique())
        
        if st.button("🎯 Predict Severity", type="primary"):
            # Create prediction dataframe
            pred_data = pd.DataFrame({
                'Speed_kmph': [speed],
                'Age_of_Driver': [age],
                'Location': [location],
                'Weather_Condition': [weather],
                'Road_Condition': [road],
                'Light_Condition': [light],
                'Vehicle_Type': [vehicle]
            })
            
            # Make prediction
            prediction = clf_pipeline.predict(pred_data)[0]
            prediction_proba = clf_pipeline.predict_proba(pred_data)[0]
            severity_label = le.inverse_transform([prediction])[0]
            
            # Display prediction
            st.markdown("### Prediction Result:")
            
            if severity_label.lower() == 'fatal':
                st.error(f"⚠️ **Predicted Severity: {severity_label}**")
            elif severity_label.lower() == 'serious':
                st.warning(f"⚠️ **Predicted Severity: {severity_label}**")
            else:
                st.info(f"ℹ️ **Predicted Severity: {severity_label}**")
            
            # Show probabilities
            st.write("**Confidence Scores:**")
            prob_df = pd.DataFrame({
                'Severity': le.inverse_transform(range(len(prediction_proba))),
                'Probability': prediction_proba
            }).sort_values('Probability', ascending=False)
            
            for idx, row in prob_df.iterrows():
                st.progress(row['Probability'], text=f"{row['Severity']}: {row['Probability']:.2%}")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file has the correct format with columns: Accident_ID, Speed_kmph, Age_of_Driver, Location, Weather_Condition, Road_Condition, Light_Condition, Vehicle_Type, Severity")

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to begin")
    
    st.markdown("""
    ### 📋 Required CSV Format:
    Your CSV file should contain the following columns:
    - **Accident_ID**: Unique identifier
    - **Speed_kmph**: Speed in kilometers per hour
    - **Age_of_Driver**: Driver's age
    - **Location**: Accident location
    - **Weather_Condition**: Weather conditions
    - **Road_Condition**: Road conditions
    - **Light_Condition**: Lighting conditions
    - **Vehicle_Type**: Type of vehicle
    - **Severity**: Accident severity (e.g., Minor, Serious, Fatal)
    
    ### 🚀 Features:
    - ✅ Automated data preprocessing
    - ✅ Random Forest classification model
    - ✅ Feature importance visualization
    - ✅ Real-time severity prediction
    - ✅ Model accuracy metrics
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Traffic Accident Severity Prediction System")