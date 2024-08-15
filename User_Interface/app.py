import streamlit as st
import sqlalchemy
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import joblib  # type: ignore 
import os
import streamlit.components.v1 as components
import time


import nltk # type: ignore
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Importing the feature computation function
from feature_module import compute_features

# Database setup using SQLAlchemy with SQLite
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    username = Column(String, primary_key=True)
    password = Column(String)

# Create an SQLite database file named 'users.db'
engine = create_engine('sqlite:///users.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Loading the model using joblib
@st.cache_resource
def load_model():
    # Get the directory where the current script is located
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the full path to the model file
    model_path = os.path.join(dir_path, 'human_AI_classifier_model.joblib')
    
    # Load the model
    model = joblib.load(model_path)
    return model

model = load_model()

# Sign-up page
def sign_up():
    st.title("Sign Up")
    st.write("Please create an account to use the AI Text Detector.")

    # Sign-up form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if username and password:
            if password == confirm_password:
                # Hashing the password
                hashed_password = hash_password(password)
                # Checking if the user already exists
                if db_session.query(User).filter_by(username=username).first() is not None:
                    st.error("Username already exists. Please choose another.")
                else:
                    # Store the new user
                    new_user = User(username=username, password=hashed_password)
                    db_session.add(new_user)
                    db_session.commit()
                    st.success("Account created successfully! Please log in.")
                    st.session_state.authenticated = False  # Redirect to login after sign up
            else:
                st.error("Passwords do not match. Please try again.")
        else:
            st.error("Please fill in all fields.")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login page
def login():
    st.title("Login")
    st.write("Please log in to access the AI Text Detector.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    def on_login():
        hashed_password = hash_password(password)
        user = db_session.query(User).filter_by(username=username, password=hashed_password).first()
        if user:
            st.session_state.authenticated = True
            st.session_state.username = username
        else:
            st.error("Incorrect username or password. Please try again.")

    st.button("Login", on_click=on_login)


# Logout function
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.success("Logged out successfully!")


# Main text detector app
def ai_text_detector():
    st.title("🔍 AI Text Detector")

    st.write(
        """
        Welcome to the AI Text Detector! Enter text below to classify it as AI-written or Human-written.
        """
    )

    # Sidebar for options
    st.sidebar.header("Options")
    st.sidebar.markdown("Use this sidebar to select example text or enter your own.")

    # Provide example text options
    example_text_options = {
        "Enter your own text": "",
        "Example 1": "Once upon a time, a curious young fox named Fenna, who lived in a lush, vibrant forest, longed to explore the world beyond its leafy borders. Inspired by stories told by Oran, a wise old owl, about distant lands filled with magical creatures and towering cities, Fenna set out on an adventure, fueled by a promise to return with tales of her own. Along her journey, she befriended a playful rabbit named Riko and a thoughtful deer named Dela, facing challenges like fierce storms and treacherous paths. Despite these obstacles, the trio used gifts from forest friends—a sharp stone, a soft feather, and nourishing berries—to overcome adversity and deepen their bonds. Eventually, Fenna returned home, her spirit enriched with stories of courage, friendship, and the beauty of exploration, reminding her forest companions that sometimes, the journey itself is the greatest story of all.",
        "Example 2": "Natural Language Processing (NLP) revolutionizes human-computer interaction, bridging the gap between human communication and machine understanding. By leveraging computational linguistics and artificial intelligence, NLP enables computers to comprehend, interpret, and generate human language in a meaningful way. From virtual assistants like Siri and Alexa to language translation tools and sentiment analysis algorithms, NLP applications permeate daily life, powering search engines, social media platforms, and customer service interactions. Its significance extends across industries, from healthcare and finance to education and entertainment. As NLP continues to advance, unlocking deeper insights from vast amounts of textual data, its impact on society grows exponentially, promising a future where machines truly understand human language.",
    }

    # Initialize session state for text input
    # Set or update the session state before creating the widget
    if "text_input" not in st.session_state:
        st.session_state.text_input = example_text_options["Enter your own text"]

    # Select example text
    example_selection = st.sidebar.selectbox("Choose an example text", list(example_text_options.keys()))

    # Update text area only if the example text changes
    if st.sidebar.button("Use Example Text"):
        st.session_state.text_input = example_text_options[example_selection]

    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        st.sidebar.write(f"Logged in as: {st.session_state.username}")

    # text area for input
    st.subheader("Enter Text")
    text_input = st.text_area("Type your text here:", height=400, value=st.session_state.text_input, key="text_input")

    # Button to trigger prediction
    if st.button("Predict"):
        # Prediction section
        st.subheader("Prediction")
        
        if text_input.strip():  # Ensure text is not just whitespace
            try:
                # Compute features for the input text
                features_df = compute_features(text_input)

                # Predict class and probabilities
                predicted_class = model.predict(features_df)
                prediction_probabilities = model.predict_proba(features_df)

                # Display results with progress bars
                predicted_label = "AI-written" if predicted_class[0] == 1 else "Human-written"
                st.markdown(f"**Predicted Class:** {predicted_label}")

                # Show probability as progress bars and percentage
                st.progress(prediction_probabilities[0][1])
                st.write(f"AI-written: {prediction_probabilities[0][1] * 100:.2f}%")

                st.progress(prediction_probabilities[0][0])
                st.write(f"Human-written: {prediction_probabilities[0][0] * 100:.2f}%")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text to predict.")

    

    #Button for logout     
    if st.sidebar.button("Logout", on_click=logout):
        pass  # The on_click will handle the logout action


# Main navigation
if st.session_state.authenticated:
    ai_text_detector()
else:
    choice = st.sidebar.radio("Select Action", ["Sign Up", "Login"])
    if choice == "Sign Up":
        sign_up()
    elif choice == "Login":
        login()

