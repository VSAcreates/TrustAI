import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import time
import re
import string
import logging
import warnings
import sounddevice as sd
import soundfile as sf
import queue
import keyboard
from datetime import datetime
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class FraudDetectionSystem:
    def __init__(self, data_path='pages/fraud_call_data.csv'):
        self.data_path = data_path
        self.vectorizer = None
        self.model = None
        self.safe_phrases = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            # Load the dataset
            data = pd.read_csv(self.data_path)
            
            # Fix: Check if the first row contains column names
            if 'Id' in data.columns and 'transcript' in data.columns and 'is_fraud' in data.columns:
                # Data loaded correctly
                pass
            elif data.shape[1] == 3 and isinstance(data.iloc[0, 0], str) and data.iloc[0, 0] == 'Id':
                # The first row is duplicated column names, drop it
                data = data.iloc[1:].reset_index(drop=True)
                data.columns = ['Id', 'transcript', 'is_fraud']
            
            # Ensure is_fraud is numeric
            data['is_fraud'] = pd.to_numeric(data['is_fraud'], errors='coerce').fillna(0).astype(int)
            
            # Create safe examples list
            self.safe_phrases = data[data['is_fraud'] == 0]['transcript'].tolist()
            
            # If no safe examples in dataset, add some programmatically
            if len(self.safe_phrases) == 0:
                self.safe_phrases = [
                    "hello how are you doing today",
                    "just calling to check on you",
                    "what are your plans for the weekend",
                    "the weather is nice today isn't it",
                    "can we meet for coffee tomorrow",
                    "I wanted to ask about your family",
                    "how was your day at work",
                    "did you see the game last night",
                    "what time should we meet",
                    "thanks for calling me back"
                ]
                
                # Convert to DataFrame and concatenate
                safe_df = pd.DataFrame({
                    'Id': [f'n{i+1}' for i in range(len(self.safe_phrases))],
                    'transcript': self.safe_phrases,
                    'is_fraud': [0] * len(self.safe_phrases)
                })
                
                # Combine with original data
                data = pd.concat([data, safe_df], ignore_index=True)
            
            # Extract features and labels
            X = data['transcript']
            y = data['is_fraud']
            
            # Preprocess all text
            X = X.apply(self.preprocess_text)
            
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_model(self):
        """Train the fraud detection model"""
        try:
            # Load and preprocess data
            X, y = self.load_data()
            
            # Vectorize the text data with enhanced parameters
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2
            )
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y
            )
            
            # Train multiple models and select the best one
            models = {
                'LogisticRegression': LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    C=1.0,
                    solver='liblinear'
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42
                )
            }
            
            best_accuracy = 0
            best_model_name = None
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    self.model = model
            
            logger.info(f"Selected model: {best_model_name} with accuracy: {best_accuracy:.4f}")
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def detect_fraud(self, text):
        """Detect fraud in text with confidence score"""
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return False, 0.0
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Basic pattern matching for common fraud phrases
            common_fraud_patterns = [
                "otp", "verification code", "urgent", "police", "arrest", 
                "court", "warrant", "freeze", "block", "investigation", 
                "credit card details", "account details", "password", "pin", 
                "money transfer", "verification fee", "tax payment"
            ]
            
            # Check for exact matches with known fraud phrases
            if any(pattern in processed_text for pattern in common_fraud_patterns):
                direct_match_boost = 0.2
            else:
                direct_match_boost = 0.0
            
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([processed_text])
            
            # Get prediction probability
            proba = self.model.predict_proba(text_vectorized)[0][1]
            
            # Apply the pattern match boost (capped at 0.95)
            adjusted_proba = min(0.95, proba + direct_match_boost)
            
            # Use a threshold (0.65 = 65% confidence)
            is_fraud = adjusted_proba >= 0.65
            
            return is_fraud, adjusted_proba
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            return False, 0.0

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recording_queue = queue.Queue()
        self.is_recording = False
        self.samplerate = 44100
    
    def record_audio_stream(self, filename, max_duration=300):
        """Record audio from microphone with dynamic duration control"""
        def audio_callback(indata, frames, time, status):
            """Callback for audio recording"""
            if status:
                logger.warning(f"Audio status: {status}")
            self.recording_queue.put(indata.copy())
        
        def save_audio():
            """Save recorded audio to file"""
            try:
                logger.info(f"Starting recording to {filename}...")
                
                # Set up the recording stream
                with sd.InputStream(samplerate=self.samplerate, channels=1, callback=audio_callback):
                    # Create an empty WAV file
                    with sf.SoundFile(filename, mode='w', samplerate=self.samplerate,
                                     channels=1, subtype='PCM_16') as file:
                        start_time = time.time()
                        self.is_recording = True
                        
                        # Record until ESC is pressed or max duration reached
                        while self.is_recording and (time.time() - start_time) < max_duration:
                            if keyboard.is_pressed('esc'):
                                self.is_recording = False
                                break
                                
                            # Get data from queue
                            try:
                                data = self.recording_queue.get(timeout=0.5)
                                file.write(data)
                            except queue.Empty:
                                pass
                        
                        # If max duration reached
                        if (time.time() - start_time) >= max_duration:
                            self.is_recording = False
                
                logger.info(f"Recording saved as {filename}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving recording: {e}")
                return False
        
        try:
            # Start recording
            self.is_recording = True
            save_audio()
            return True
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return False
    
    def transcribe_audio(self, filename):
        """Transcribe audio file to text"""
        try:
            with sr.AudioFile(filename) as source:
                audio = self.recognizer.record(source)
                
                # Try multiple language options for better accuracy
                languages = ['en-IN', 'en-US', 'en-GB']
                
                for lang in languages:
                    try:
                        text = self.recognizer.recognize_google(audio, language=lang)
                        logger.info(f"Transcribed text ({lang}): {text}")
                        return text
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        logger.error(f"Google API error: {e}")
                        break
                
                logger.warning("Could not understand audio")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

class FraudAlertSystem:
    def __init__(self):
        self.fraud_alert_file = "fraud_alert.mp3"
        self.safe_alert_file = "safe_alert.mp3"
        self.initialize_audio_files()
    
    def initialize_audio_files(self):
        """Initialize alert sound files"""
        try:
            # Create fraud alert file if it doesn't exist
            if not os.path.exists(self.fraud_alert_file):
                tts = gTTS(text="Warning! Potential fraud detected!", lang='en', slow=False)
                tts.save(self.fraud_alert_file)
            
            # Create safe alert file if it doesn't exist
            if not os.path.exists(self.safe_alert_file):
                tts = gTTS(text="This call appears to be safe.", lang='en', slow=False)
                tts.save(self.safe_alert_file)
        except Exception as e:
            logger.error(f"Error creating alert files: {e}")
    
    def generate_warning(self, confidence, lang='en'):
        """Generate warning message based on confidence level"""
        if confidence >= 0.8:
            severity = "VERY HIGH"
            message = (
                f"WARNING: {severity} probability of fraud detected ({confidence:.0%} confidence). "
                "DO NOT share any personal information, financial details, or make any payments. "
                "Hang up immediately and report this call to the authorities."
            )
        elif confidence >= 0.65:
            severity = "HIGH"
            message = (
                f"WARNING: {severity} probability of fraud detected ({confidence:.0%} confidence). "
                "Be extremely cautious. Do not share personal or financial information. "
                "Consider ending the call."
            )
        else:
            severity = "MODERATE"
            message = (
                f"CAUTION: {severity} probability of fraud detected ({confidence:.0%} confidence). "
                "Be cautious and do not share sensitive information."
            )
        
        # Translate the message if needed
        if lang != 'en':
            try:
                translator = GoogleTranslator(source='en', target=lang)
                message = translator.translate(message)
            except Exception as e:
                logger.error(f"Translation error: {e}")
        
        return message
    
    def generate_safe_message(self, confidence, lang='en'):
        """Generate safe message based on confidence level"""
        message = (
            f"This call appears to be safe ({(1-confidence):.0%} confidence). "
            "However, always remain cautious when sharing personal information."
        )
        
        # Translate the message if needed
        if lang != 'en':
            try:
                translator = GoogleTranslator(source='en', target=lang)
                message = translator.translate(message)
            except Exception as e:
                logger.error(f"Translation error: {e}")
        
        return message

def detect_language(text):
    """Detect language of the text"""
    try:
        # Use deep_translator for language detection
        translator = GoogleTranslator(source='auto', target='en')
        detected_lang = translator.source
        return detected_lang
    except:
        return 'en'  # Default to English

def autoplay_audio(file_path):
    """Autoplay audio in Streamlit"""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Fraud Call Detection System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fraud Call Detection System")
    st.markdown("---")
    
    # Initialize components
    if 'fraud_system' not in st.session_state:
        with st.spinner("Initializing fraud detection system..."):
            st.session_state.fraud_system = FraudDetectionSystem()
            accuracy = st.session_state.fraud_system.train_model()
            st.session_state.audio_processor = AudioProcessor()
            st.session_state.alert_system = FraudAlertSystem()
            st.success(f"System initialized with model accuracy: {accuracy:.2%}")
    
    # Sidebar
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Option", ["Live Call Analysis", "Text Analysis", "Audio File Analysis"])
    
    if option == "Live Call Analysis":
        st.header("Live Call Analysis")
        st.warning("Note: This feature requires microphone access and may not work in all browsers.")
        
        if st.button("Start Recording (Press ESC to stop)"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"call_recording_{timestamp}.wav"
            
            with st.spinner("Recording in progress..."):
                if st.session_state.audio_processor.record_audio_stream(audio_file):
                    st.success("Recording completed!")
                    
                    with st.spinner("Transcribing audio..."):
                        text = st.session_state.audio_processor.transcribe_audio(audio_file)
                        
                        if text:
                            st.subheader("Transcript")
                            st.text_area("Call Transcript", text, height=150)
                            
                            # Detect language
                            detected_lang = detect_language(text)
                            
                            # Detect fraud
                            is_fraud, confidence = st.session_state.fraud_system.detect_fraud(text)
                            
                            if is_fraud:
                                st.error(f"🚨 FRAUD DETECTED ({confidence:.0%} confidence)")
                                warning = st.session_state.alert_system.generate_warning(confidence, detected_lang)
                                st.warning(warning)
                                
                                # Play fraud alert
                                autoplay_audio(st.session_state.alert_system.fraud_alert_file)
                            else:
                                st.success(f"✅ Call appears safe ({(1-confidence):.0%} confidence)")
                                safe_message = st.session_state.alert_system.generate_safe_message(confidence, detected_lang)
                                st.info(safe_message)
                                
                                # Play safe alert
                                autoplay_audio(st.session_state.alert_system.safe_alert_file)
                        else:
                            st.error("Could not transcribe audio. Please try again.")
    
    elif option == "Text Analysis":
        st.header("Text Analysis")
        text_input = st.text_area("Enter call transcript for analysis:", height=150)
        
        if st.button("Analyze Text"):
            if text_input:
                is_fraud, confidence = st.session_state.fraud_system.detect_fraud(text_input)
                
                if is_fraud:
                    st.error(f"🚨 FRAUD DETECTED ({confidence:.0%} confidence)")
                    warning = st.session_state.alert_system.generate_warning(confidence)
                    st.warning(warning)
                    
                    # Play fraud alert
                    autoplay_audio(st.session_state.alert_system.fraud_alert_file)
                else:
                    st.success(f"✅ Call appears safe ({(1-confidence):.0%} confidence)")
                    safe_message = st.session_state.alert_system.generate_safe_message(confidence)
                    st.info(safe_message)
                    
                    # Play safe alert
                    autoplay_audio(st.session_state.alert_system.safe_alert_file)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif option == "Audio File Analysis":
        st.header("Audio File Analysis")
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            # Save the uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"uploaded_audio_{timestamp}.wav"
            
            with open(audio_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing audio file..."):
                text = st.session_state.audio_processor.transcribe_audio(audio_file)
                
                if text:
                    st.subheader("Transcript")
                    st.text_area("Call Transcript", text, height=150)
                    
                    # Detect language
                    detected_lang = detect_language(text)
                    
                    # Detect fraud
                    is_fraud, confidence = st.session_state.fraud_system.detect_fraud(text)
                    
                    if is_fraud:
                        st.error(f"🚨 FRAUD DETECTED ({confidence:.0%} confidence)")
                        warning = st.session_state.alert_system.generate_warning(confidence, detected_lang)
                        st.warning(warning)
                        
                        # Play fraud alert
                        autoplay_audio(st.session_state.alert_system.fraud_alert_file)
                    else:
                        st.success(f"✅ Call appears safe ({(1-confidence):.0%} confidence)")
                        safe_message = st.session_state.alert_system.generate_safe_message(confidence, detected_lang)
                        st.info(safe_message)
                        
                        # Play safe alert
                        autoplay_audio(st.session_state.alert_system.safe_alert_file)
                else:
                    st.error("Could not transcribe audio. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This Fraud Call Detection System uses machine learning to analyze phone call transcripts 
    and detect potential fraudulent activity. The system can:
    - Analyze live calls (requires microphone)
    - Analyze text transcripts
    - Analyze pre-recorded audio files
    """)

if __name__ == "__main__":
    main()