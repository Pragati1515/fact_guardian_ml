# fact_guardian_ml.py - STREAMLIT CLOUD READY
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
import os

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Configure page
st.set_page_config(
    page_title="FactGuardian ML • AI-Powered Truth Verification",
    page_icon="🛡️",
    layout="wide"
)

# Get API Key from Streamlit Secrets
API_KEY = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", "AIzaSyBfzkoRUM2GFfwIuEYRuLL5wcL0DM9eGqM")

# Custom CSS
st.markdown("""
<style>
    .guardian-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .model-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

class MLFactChecker:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.is_trained = False
        
    def create_training_data(self):
        """Create comprehensive training data for fact-checking"""
        true_claims = [
            "COVID-19 vaccines are safe and effective according to health authorities",
            "Climate change is primarily caused by human activities and greenhouse gas emissions",
            "The Earth is an oblate spheroid, not flat",
            "Smoking tobacco causes lung cancer and respiratory diseases",
            "Regular exercise improves cardiovascular health and longevity",
            "Solar energy is a renewable and sustainable power source",
            "Water boils at 100 degrees Celsius at sea level",
            "Antibiotics are effective against bacterial infections but not viruses",
            "Vaccines have eradicated diseases like smallpox and reduced polio",
            "The Great Barrier Reef is suffering from coral bleaching due to climate change",
            "Mental health is as important as physical health",
            "Recycling reduces landfill waste and conserves natural resources",
            "NASA landed astronauts on the Moon during the Apollo missions",
            "The human body needs vitamins and minerals for proper functioning",
            "Electric vehicles produce zero tailpipe emissions"
        ]
        
        false_claims = [
            "Vaccines contain microchips for government tracking",
            "5G networks spread coronavirus and other diseases",
            "The Earth is flat and stationary in space",
            "Chemtrails from airplanes are used for population control",
            "The Moon landing in 1969 was completely faked in a studio",
            "Vitamin C alone can cure COVID-19 infection",
            "Cancer can be cured by drinking baking soda solutions",
            "Humans only use 10% of their brain capacity",
            "Microwave ovens cause cancer through radiation leaks",
            "Global warming is a hoax created by scientists for funding",
            "Genetically modified foods are inherently dangerous to eat",
            "HIV does not cause AIDS",
            "The COVID-19 pandemic was planned by world governments",
            "Face masks cause oxygen deprivation and carbon dioxide poisoning",
            "Wind turbines cause cancer from noise pollution"
        ]
        
        claims = true_claims + false_claims
        labels = [1] * len(true_claims) + [0] * len(false_claims)  # 1=True, 0=False
        
        return claims, labels
    
    def train_models(self):
        """Train all 5 ML models with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create training data
        claims, labels = self.create_training_data()
        
        # Feature engineering
        status_text.text("🔧 Extracting features from training data...")
        X = self.vectorizer.fit_transform(claims)
        y = np.array(labels)
        progress_bar.progress(20)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_results = {}
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"🤖 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=3)  # Reduced for speed
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'predictions': y_pred
            }
            
            progress = 20 + ((i + 1) / len(self.models)) * 60
            progress_bar.progress(int(progress))
        
        progress_bar.progress(100)
        status_text.text("✅ All models trained successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        self.is_trained = True
        self.model_results = model_results
        return model_results
    
    def predict_claim(self, claim_text):
        """Predict truthfulness using all trained models"""
        if not self.is_trained:
            return None
            
        # Transform input
        X_input = self.vectorizer.transform([claim_text])
        
        predictions = {}
        confidence_scores = {}
        
        for name, result in self.model_results.items():
            model = result['model']
            
            # Get prediction and probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_input)[0]
                prediction = model.predict(X_input)[0]
                confidence = max(proba)
            else:
                prediction = model.predict(X_input)[0]
                confidence = 0.5
                
            predictions[name] = prediction
            confidence_scores[name] = confidence
        
        return predictions, confidence_scores

def get_fact_check_results(query):
    """Fetch fact-check results from Google API"""
    if not API_KEY:
        st.error("❌ API key not configured. Please check your secrets.toml file.")
        return []
        
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    
    try:
        with st.spinner("🔍 Searching verified fact-checking sources..."):
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                st.error(f"API Error: {response.status_code}")
                return []
                
            data = response.json()
        
        results = []
        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                results.append({
                    "publisher": review.get("publisher", {}).get("name", "Unknown Source"),
                    "rating": review.get("textualRating", "No Rating"),
                    "url": review.get("url", ""),
                    "title": review.get("title", "No Title Available"),
                    "claim_date": claim.get("claimDate", "")
                })
        return results
    except Exception as e:
        st.error(f"Network error: {str(e)}")
        return []

def main():
    st.markdown("""
    <div class="guardian-header">
        <h1>🛡️ FactGuardian ML Pro</h1>
        <h3>5-Model Machine Learning Fact Verification System</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize ML system
    if 'ml_checker' not in st.session_state:
        st.session_state.ml_checker = MLFactChecker()
    
    ml_checker = st.session_state.ml_checker
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.info("**API Status:** " + ("✅ Connected" if API_KEY else "❌ Not Configured"))
        
        st.markdown("### 🤖 ML Models")
        st.write("""
        **Trained Models:**
        - Random Forest
        - Support Vector Machine  
        - Logistic Regression
        - Naive Bayes
        - Decision Tree
        """)
        
        if st.button("🔄 Retrain Models"):
            st.session_state.ml_checker.train_models()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Claim Verification")
        user_claim = st.text_area(
            "Enter a claim or statement to verify:",
            placeholder="Example: 'Climate change is a hoax created by scientists'",
            height=120
        )
        
        if st.button("🚀 Verify with 5 ML Models", type="primary", use_container_width=True):
            if user_claim.strip():
                # Train models if not already trained
                if not ml_checker.is_trained:
                    with st.spinner("Initializing AI models for the first time..."):
                        ml_checker.train_models()
                
                # Store current claim
                st.session_state.current_claim = user_claim
                st.rerun()
            else:
                st.warning("Please enter a claim to verify.")
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if ml_checker.is_trained:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Models Ready", "5/5", "✅")
            with col2_2:
                avg_accuracy = np.mean([result['accuracy'] for result in ml_checker.model_results.values()])
                st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        else:
            st.info("Models not trained yet. Enter a claim to begin.")
    
    # Display results if available
    if hasattr(st.session_state, 'current_claim') and st.session_state.current_claim:
        user_claim = st.session_state.current_claim
        
        st.markdown("---")
        st.markdown(f"### 📝 Analyzing: *\"{user_claim}\"*")
        
        # Get predictions
        ml_predictions, confidences = ml_checker.predict_claim(user_claim)
        fact_checks = get_fact_check_results(user_claim)
        
        # Display ML Results
        st.markdown("#### 🤖 ML Model Predictions")
        
        results_data = []
        for model_name, prediction in ml_predictions.items():
            confidence = confidences[model_name]
            verdict = "✅ TRUE" if prediction == 1 else "❌ FALSE"
            accuracy = ml_checker.model_results[model_name]['accuracy']
            
            results_data.append({
                'Model': model_name,
                'Verdict': verdict,
                'Confidence': f"{confidence:.1%}",
                'Accuracy': f"{accuracy:.1%}"
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Overall consensus
        true_count = sum(1 for p in ml_predictions.values() if p == 1)
        false_count = sum(1 for p in ml_predictions.values() if p == 0)
        consensus = "TRUE" if true_count > false_count else "FALSE"
        
        st.markdown(f"#### 🎯 Overall ML Consensus: **{consensus}** ({true_count}-{false_count})")
        
        # Model performance chart
        st.markdown("#### 📈 Model Performance Comparison")
        model_names = list(ml_checker.model_results.keys())
        accuracies = [ml_checker.model_results[name]['accuracy'] for name in model_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(model_names, accuracies, color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe'])
        ax.set_xlabel('Accuracy Score')
        ax.set_title('ML Model Training Performance')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, accuracy in zip(bars, accuracies):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{accuracy:.3f}', va='center')
        
        st.pyplot(fig)
        
        # Fact check results
        st.markdown("#### 📰 External Fact Check Results")
        if fact_checks:
            st.success(f"✅ Found {len(fact_checks)} external verification(s)")
            for i, check in enumerate(fact_checks[:3]):  # Show top 3
                rating = check['rating'].lower()
                border_color = "#00C853" if 'true' in rating else "#FF1744" if 'false' in rating else "#FF9800"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {border_color}; padding: 15px; background: white; margin: 10px 0; border-radius: 5px;">
                    <h4 style="margin-top: 0;">{check['title']}</h4>
                    <p><strong>Source:</strong> 📰 {check['publisher']}</p>
                    <p><strong>Rating:</strong> <span style="color: {border_color}; font-weight: bold;">{check['rating']}</span></p>
                    <p><a href="{check['url']}" target="_blank" style="color: #667eea; text-decoration: none;">🔗 Read Full Analysis →</a></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No external fact-check results found. This claim may be new or use uncommon phrasing.")

if __name__ == "__main__":
    main()