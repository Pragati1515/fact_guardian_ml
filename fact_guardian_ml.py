# fact_guardian_ml.py - FIXED CLAIM VERIFICATION
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

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
    .fact-check-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .rating-true {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .rating-false {
        background: linear-gradient(135deg, #f44336, #da190b);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .rating-mixed {
        background: linear-gradient(135deg, #FF9800, #f57c00);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .rating-unknown {
        background: linear-gradient(135deg, #757575, #616161);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .publisher-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
    .result-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class MLFactChecker:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
            "Support Vector Machine": SVC(kernel='linear', probability=True, C=1.0, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, C=0.7),
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42)
        }
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        self.is_trained = False
        self.model_results = {}
        
    def create_training_data(self):
        """Create comprehensive training data for fact-checking"""
        true_claims = [
            "COVID-19 vaccines are safe and effective according to global health authorities",
            "Climate change is primarily caused by human activities and greenhouse gas emissions",
            "The Earth is an oblate spheroid that orbits the Sun, not flat",
            "Smoking tobacco significantly increases the risk of lung cancer and heart disease",
            "Regular physical exercise improves cardiovascular health and extends lifespan",
            "Solar energy is a renewable and sustainable power source that reduces carbon emissions",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure",
            "Antibiotics are effective against bacterial infections but do not work on viruses",
            "Vaccines have successfully eradicated smallpox and dramatically reduced polio cases worldwide",
            "The Great Barrier Reef is experiencing coral bleaching due to rising ocean temperatures",
            "Mental health disorders are real medical conditions that require proper treatment",
            "Recycling programs help reduce landfill waste and conserve natural resources",
            "NASA successfully landed astronauts on the Moon during six Apollo missions",
            "The human body requires essential vitamins and minerals for optimal health",
            "Electric vehicles produce zero direct tailpipe emissions during operation"
        ]
        
        false_claims = [
            "Vaccines contain microchips for government tracking and population control",
            "5G cellular networks spread coronavirus and cause other serious diseases",
            "The Earth is flat and stationary, with Antarctica as an ice wall surrounding the edges",
            "Chemtrails from airplanes contain chemicals for mind control and weather manipulation",
            "The Moon landing in 1969 was completely faked in a Hollywood film studio",
            "Vitamin C and zinc supplements alone can cure COVID-19 infection completely",
            "Cancer can be cured by drinking baking soda solutions and avoiding conventional treatment",
            "Humans only use 10% of their brain capacity according to scientific studies",
            "Microwave ovens cause cancer through dangerous radiation leaks during operation",
            "Global warming is a hoax created by scientists to secure research funding",
            "Genetically modified foods are inherently dangerous and cause numerous health problems",
            "HIV does not cause AIDS and the connection is a pharmaceutical conspiracy",
            "The COVID-19 pandemic was intentionally planned and released by world governments",
            "Face masks cause oxygen deprivation and dangerous carbon dioxide poisoning",
            "Wind turbines cause cancer through low-frequency noise and vibration pollution"
        ]
        
        claims = true_claims + false_claims
        labels = [1] * len(true_claims) + [0] * len(false_claims)
        
        return claims, labels
    
    def train_models(self):
        """Train all 5 ML models"""
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"🤖 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            
            self.model_results[name] = {
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
        return self.model_results
    
    def predict_claim(self, claim_text):
        """Predict truthfulness using all trained models"""
        if not self.is_trained:
            st.error("❌ Models not trained yet. Please train models first.")
            return None, None
            
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
                # For SVM without probability
                prediction = model.predict(X_input)[0]
                confidence = 0.7  # Default confidence for non-probability models
                
            predictions[name] = prediction
            confidence_scores[name] = confidence
        
        return predictions, confidence_scores

def get_fact_check_results(query):
    """Fetch fact-check results from Google API - FIXED VERSION"""
    if not API_KEY:
        st.error("❌ API key not configured. Please check your secrets.toml file.")
        return []
        
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": API_KEY,
        "languageCode": "en"
    }
    
    try:
        with st.spinner("🔍 Searching 100+ verified fact-checking sources..."):
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code != 200:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return []
                
            data = response.json()
        
        results = []
        if "claims" in data and data["claims"]:
            for claim in data.get("claims", []):
                claim_text = claim.get("text", "")
                
                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {}).get("name", "Unknown Source")
                    rating = review.get("textualRating", "No Rating")
                    url = review.get("url", "")
                    title = review.get("title", "No Title Available")
                    
                    results.append({
                        "publisher": publisher,
                        "rating": rating,
                        "url": url,
                        "title": title,
                        "original_claim": claim_text,
                        "claim_date": claim.get("claimDate", "")
                    })
            
            st.success(f"✅ Found {len(results)} fact-check results")
        else:
            st.warning("⚠️ No direct fact-checks found for this claim")
            
        return results
        
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return []

def get_rating_class(rating):
    """Get CSS class for different rating types"""
    rating_lower = str(rating).lower()
    if 'true' in rating_lower and 'false' not in rating_lower and 'mostly' not in rating_lower:
        return 'rating-true'
    elif 'false' in rating_lower:
        return 'rating-false'
    elif 'mix' in rating_lower or 'half' in rating_lower or 'mostly' in rating_lower:
        return 'rating-mixed'
    else:
        return 'rating-unknown'

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
        st.session_state.models_trained = False
    
    ml_checker = st.session_state.ml_checker
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.info(f"**API Status:** {'✅ Connected' if API_KEY else '❌ Not Configured'}")
        
        if API_KEY:
            # Test API connection
            if st.button("🔍 Test API Connection", use_container_width=True):
                with st.spinner("Testing API connection..."):
                    test_results = get_fact_check_results("COVID vaccines contain microchips")
                    if test_results:
                        st.success("✅ API connection successful!")
                    else:
                        st.error("❌ API connection failed")
        
        st.markdown("### 🤖 ML Models")
        st.write("""
        **5 Ensemble Models:**
        - Random Forest
        - Support Vector Machine  
        - Logistic Regression
        - Naive Bayes
        - Decision Tree
        """)
        
        if st.button("🔄 Train Models First", use_container_width=True, type="primary"):
            with st.spinner("Training all 5 ML models..."):
                ml_checker.train_models()
                st.session_state.models_trained = True
            st.success("✅ Models trained successfully!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Claim Verification")
        
        # Quick test claims
        st.markdown("**Quick Test Claims:**")
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("COVID Microchips", use_container_width=True):
                st.session_state.test_claim = "COVID-19 vaccines contain microchips for government tracking"
            if st.button("Flat Earth", use_container_width=True):
                st.session_state.test_claim = "The Earth is flat and NASA is lying"
                
        with quick_col2:
            if st.button("5G Coronavirus", use_container_width=True):
                st.session_state.test_claim = "5G networks spread coronavirus"
            if st.button("Climate Hoax", use_container_width=True):
                st.session_state.test_claim = "Climate change is a hoax created by scientists"
        
        # Claim input
        user_claim = st.text_area(
            "Or enter your own claim to verify:",
            value=st.session_state.get('test_claim', ''),
            placeholder="Example: 'COVID-19 vaccines contain microchips for government tracking'",
            height=100,
            key="claim_input"
        )
        
        verify_disabled = not ml_checker.is_trained
        
        if st.button("🚀 Verify Claim", 
                    type="primary", 
                    use_container_width=True,
                    disabled=verify_disabled):
            
            if not user_claim.strip():
                st.error("❌ Please enter a claim to verify")
            elif not ml_checker.is_trained:
                st.error("❌ Please train models first using the button in sidebar")
            else:
                # Store current claim
                st.session_state.current_claim = user_claim
                st.session_state.show_results = True
                st.rerun()
    
    with col2:
        st.markdown("### 📊 System Status")
        if ml_checker.is_trained:
            st.success("✅ Models Trained & Ready")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Models Ready", "5/5")
            with col2_2:
                avg_accuracy = np.mean([result['accuracy'] for result in ml_checker.model_results.values()])
                st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        else:
            st.error("❌ Models Not Trained")
            st.info("Click 'Train Models First' in sidebar")
    
    # Display results if available
    if st.session_state.get('show_results', False) and st.session_state.get('current_claim'):
        user_claim = st.session_state.current_claim
        
        st.markdown("---")
        st.markdown(f"### 📝 Analyzing: *\"{user_claim}\"*")
        
        # Create tabs for different result types
        tab1, tab2 = st.tabs(["🤖 ML Model Analysis", "📰 Fact Check Results"])
        
        with tab1:
            st.markdown("#### Machine Learning Predictions")
            
            # Get ML predictions
            with st.spinner("🧠 Analyzing with AI models..."):
                ml_predictions, confidences = ml_checker.predict_claim(user_claim)
            
            if ml_predictions:
                # Display ML Results
                results_data = []
                for model_name, prediction in ml_predictions.items():
                    confidence = confidences[model_name]
                    verdict = "✅ TRUE" if prediction == 1 else "❌ FALSE"
                    accuracy = ml_checker.model_results[model_name]['accuracy']
                    
                    results_data.append({
                        'Model': model_name,
                        'Verdict': verdict,
                        'Confidence': f"{confidence:.1%}",
                        'Training Accuracy': f"{accuracy:.1%}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Overall consensus
                true_count = sum(1 for p in ml_predictions.values() if p == 1)
                false_count = sum(1 for p in ml_predictions.values() if p == 0)
                consensus = "TRUE" if true_count > false_count else "FALSE"
                
                st.markdown(f"#### 🎯 Overall ML Consensus: **{consensus}** ({true_count}-{false_count})")
        
        with tab2:
            st.markdown("#### External Fact Check Verification")
            
            # Get fact check results
            fact_checks = get_fact_check_results(user_claim)
            
            if fact_checks:
                st.success(f"✅ Found {len(fact_checks)} verified fact-check(s)")
                
                for i, check in enumerate(fact_checks[:4]):  # Show top 4
                    rating_class = get_rating_class(check['rating'])
                    
                    st.markdown(f"""
                    <div class="fact-check-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #333; flex: 1;">{check['title']}</h4>
                            <span class="{rating_class}">{check['rating']}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                            <span class="publisher-badge">📰 {check['publisher']}</span>
                            {f"<small style='color: #666;'>📅 {check['claim_date']}</small>" if check['claim_date'] else ""}
                        </div>
                        <p style="color: #666; font-style: italic; margin-bottom: 12px;">"{check['original_claim'][:200]}..."</p>
                        <a href="{check['url']}" target="_blank" style="
                            display: inline-block; 
                            background: linear-gradient(135deg, #667eea, #764ba2); 
                            color: white; 
                            padding: 8px 16px; 
                            border-radius: 20px; 
                            text-decoration: none; 
                            font-weight: 600;
                            font-size: 0.9em;
                        ">📖 Read Full Analysis →</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("""
                **No external fact-check results found for this specific claim.**
                
                This could mean:
                - The claim is very new and hasn't been fact-checked yet
                - The phrasing is unique or highly specific
                - Try rephrasing your claim with more common terminology
                
                *The ML models above are still providing AI-powered analysis based on training data.*
                """)

if __name__ == "__main__":
    main()
