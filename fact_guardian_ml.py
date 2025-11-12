# fact_guardian_ml.py - WITH MANUAL CSV UPLOAD
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
import io

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
    .upload-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class DataManager:
    """Manages training data - manual upload or synthetic"""
    
    def __init__(self):
        self.uploaded_data = None
    
    def load_uploaded_data(self, uploaded_file):
        """Load data from uploaded CSV file"""
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} records from uploaded file")
                return df
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
        return None
    
    def validate_data(self, df):
        """Validate uploaded data has required columns"""
        required_columns = ['statement', 'rating']
        if all(col in df.columns for col in required_columns):
            return True
        else:
            st.error(f"❌ Uploaded file must contain columns: {required_columns}")
            return False
    
    def map_ratings_to_labels(self, df):
        """Convert Politifact ratings to binary labels"""
        df = df.copy()
        
        # Map ratings to binary labels
        def map_rating(rating):
            rating_str = str(rating).lower()
            if any(true in rating_str for true in ['true', 'mostly true', 'half true']):
                return 1  # TRUE
            elif any(false in rating_str for false in ['false', 'pants on fire', 'mostly false']):
                return 0  # FALSE
            else:
                return -1  # UNKNOWN
        
        df['label'] = df['rating'].apply(map_rating)
        valid_data = df[df['label'] != -1]
        
        st.info(f"📊 Rating distribution: {len(valid_data[valid_data['label']==1])} True, {len(valid_data[valid_data['label']==0])} False")
        return valid_data

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
        self.data_manager = DataManager()
        
    def create_training_data(self, use_uploaded_data=False, uploaded_file=None):
        """Create training data - can use uploaded CSV or synthetic data"""
        if use_uploaded_data and uploaded_file is not None:
            # Use uploaded CSV data
            df = self.data_manager.load_uploaded_data(uploaded_file)
            if df is not None and self.data_manager.validate_data(df):
                valid_data = self.data_manager.map_ratings_to_labels(df)
                if len(valid_data) >= 10:  # Minimum data requirement
                    claims = valid_data['statement'].tolist()
                    labels = valid_data['label'].tolist()
                    st.success(f"✅ Training with {len(claims)} uploaded fact-checks")
                    return claims, labels
                else:
                    st.warning("❌ Uploaded data has insufficient valid records. Using synthetic data.")
        
        # Fallback to synthetic data
        st.info("🔄 Using enhanced synthetic training data")
        return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Enhanced synthetic training data"""
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
    
    def train_models(self, use_uploaded_data=False, uploaded_file=None):
        """Train all 5 ML models"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create training data (uploaded or synthetic)
        claims, labels = self.create_training_data(use_uploaded_data, uploaded_file)
        
        # Feature engineering
        status_text.text("🔧 Extracting features from training data...")
        X = self.vectorizer.fit_transform(claims)
        y = np.array(labels)
        progress_bar.progress(20)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        model_results = {}
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"🤖 Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
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
            
        X_input = self.vectorizer.transform([claim_text])
        predictions = {}
        confidence_scores = {}
        
        for name, result in self.data_manager.model_results.items():
            model = result['model']
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_input)[0]
                prediction = model.predict(X_input)[0]
                confidence = max(proba)
            else:
                decision = model.decision_function(X_input)[0]
                prediction = model.predict(X_input)[0]
                confidence = min(1.0, max(0.0, abs(decision) / 2.0 + 0.5))
                
            predictions[name] = prediction
            confidence_scores[name] = confidence
        
        return predictions, confidence_scores

def get_fact_check_results(query):
    """Fetch fact-check results from Google API"""
    if not API_KEY:
        st.error("❌ API key not configured.")
        return []
        
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    
    try:
        with st.spinner("🔍 Searching verified fact-checking sources..."):
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
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
    except Exception:
        return []

def main():
    st.markdown("""
    <div class="guardian-header">
        <h1>🛡️ FactGuardian ML Pro</h1>
        <h3>AI Fact Verification with Manual Data Upload & 5 ML Models</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize ML system
    if 'ml_checker' not in st.session_state:
        st.session_state.ml_checker = MLFactChecker()
    
    ml_checker = st.session_state.ml_checker
    
    # Sidebar with data upload options
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.info("**API Status:** " + ("✅ Connected" if API_KEY else "❌ Not Configured"))
        
        st.markdown("### 📁 Data Source")
        use_uploaded_data = st.checkbox("Use Uploaded CSV Data", value=False,
                                       help="Upload your own Politifact CSV file for training")
        
        uploaded_file = None
        if use_uploaded_data:
            st.markdown("""
            <div class="upload-card">
                <h4>📤 Upload Politifact CSV</h4>
                <p>File should contain:</p>
                <ul>
                    <li><code>statement</code> - The fact claim</li>
                    <li><code>rating</code> - Truth rating</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], 
                                           help="Upload your Politifact data export")
        
        if st.button("🔄 Train Models", use_container_width=True):
            with st.spinner("Training models with selected data..."):
                ml_checker.train_models(use_uploaded_data=use_uploaded_data, 
                                      uploaded_file=uploaded_file)
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Claim Verification")
        user_claim = st.text_area(
            "Enter a claim or statement to verify:",
            placeholder="Example: 'Climate change is a hoax created by scientists'",
            height=120,
            key="claim_input"
        )
        
        if st.button("🚀 Verify with 5 ML Models", type="primary", use_container_width=True):
            if user_claim.strip():
                if not ml_checker.is_trained:
                    with st.spinner("Training AI models with selected data..."):
                        ml_checker.train_models(use_uploaded_data=use_uploaded_data,
                                              uploaded_file=uploaded_file)
                st.session_state.current_claim = user_claim
                st.rerun()
            else:
                st.warning("Please enter a claim to verify.")
    
    # Display results (your existing results display code here)
    # ... [rest of your display code remains the same]

if __name__ == "__main__":
    main()
