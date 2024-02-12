import streamlit as st
import nltk
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define pairs of patterns and responses
qa_pairs = [
    ("What is your name?", "My name is Samir Sengupta."),
    ("What is your current designation?", "I currently work as an Associate Software Developer at Amigosec Synradar."),
    ("How many years of experience do you have?", "I have worked in three companies and have more than three years of experience."),
    ("What is your email address?", "My email address is Samir843301003@gmail.com."),
    ("What is your contact number?", "My contact number is +91 8356075699."),
    ("Where are you from?", "I am from Mumbai, India."),
    ("Can you provide your LinkedIn profile?", "Sure, here is my LinkedIn profile: www.linkedin.com/in/samirsengupta/"),
    ("Can you provide your GitHub profile?", "Of course, here is my GitHub profile: https://github.com/SamirSengupta"),
    ("What is your educational background?", "I hold a Bachelor of Technology (Honors) in Data Science from KES Shroff College of Information Technology, Mumbai, India. Additionally, I completed my Higher Secondary Education (Science) at Reena Mehta College of Science, Mumbai, India."),
    ("Which projects have you worked on?", "I have worked on several projects including a Movie Recommendation System, a Music Mate Song Downloading System, a Professional Portfolio, a Laptop Price Prediction application, and Customer Segmentation."),
    ("Where do you see yourself in the next 5 years?", "In the next 5 years, I envision myself becoming a proficient data scientist at a reputable company like Google, contributing to cutting-edge projects in machine learning and AI."),
    ("Do you have any experience with machine learning?", "Yes, I have extensive experience in machine learning. I have worked as a data scientist and a data analyst, progressively honing my skills in developing and deploying machine learning algorithms."),
    ("Do you have any experience with natural language processing?", "Absolutely! I have experience in natural language processing. I have created chatbots, worked on sentiment analysis, and implemented various NLP techniques in my projects."),
    ("Do you have any experience in the data science or data analysis domain?", "Yes, I have worked in the data science and data analysis domain in my previous roles. I have conducted data analysis, built predictive models, and generated actionable insights from data."),
    ("Can you provide your current CTC?", "Unfortunately, I cannot provide my current CTC as I have signed an NDA."),
    ("What is your favorite programming language?", "My favorite programming language is Python, especially when working with machine learning and AI."),
    ("Do you enjoy working in a team or individually?", "I enjoy both collaborative teamwork and working independently, adapting to the needs of the project and leveraging the strengths of each approach."),
    ("What is your approach to problem-solving?", "My approach to problem-solving involves breaking down complex problems into smaller, manageable tasks, conducting thorough research, and leveraging my analytical skills to identify effective solutions. I also value collaboration and seek input from teammates when needed."),
    ("What is your preferred method of communication?", "My preferred method of communication is email, but I am also comfortable communicating through phone calls or video conferences."),
    ("Can you provide any additional information not covered in the questions?", "Certainly! Feel free to ask about specific technologies, methodologies, or any other aspect of my background or experience that you would like to know more about."),
    ("What is your current work in your current job role?", "In my current role as an Associate Software Developer at Amigosec Synradar, I am responsible for spearheading the migration of the entire code base from PHP to Python, developing robust Python applications using Flask for proactive security measures, and implementing artificial intelligence and machine learning techniques to optimize the code base."),
    ("What did you work on in your previous roles?", "In my previous roles, I worked as a Data Analyst & Backend Developer at Profit Maxima, where I conducted data analysis on stock market data using Python and SQL, and optimized backend infrastructure with MySQL. Prior to that, I worked as a Data Scientist at Neural Thread, where I engineered GPT-2 based prompt processing engines and streamlined Neural Thread installations for enhanced user experiences."),
    ("Can you provide information about your current job and previous roles?", "In my current role at Amigosec Synradar, I am focused on migrating the code base from PHP to Python, developing robust Python applications using Flask, and implementing machine learning techniques. In my previous roles, I worked as a Data Analyst & Backend Developer at Profit Maxima, where I conducted data analysis and optimized backend infrastructure, and as a Data Scientist at Neural Thread, where I engineered prompt processing engines and streamlined installations."),
    ("What kind of projects have you worked on in your current and previous roles?", "In my current role, I have worked on projects involving code migration, Python application development, and machine learning implementation. In my previous roles, I worked on data analysis, backend optimization, and prompt processing engine development."),
    ("Can you provide an overview of your career progression?", "Certainly! I started my career as a Data Scientist at Neural Thread, where I focused on prompt processing engine development. I then transitioned to a Data Analyst & Backend Developer role at Profit Maxima, where I worked on backend optimization and data analysis. Currently, I am an Associate Software Developer at Amigosec Synradar, focusing on code migration and machine learning implementation."),
    ("What technologies do you currently work with?", "In my current role, I primarily work with Python, Flask, and machine learning libraries for code migration and application development."),
    ("Can you describe your responsibilities in your current job role?", "In my current role, I am responsible for migrating the code base from PHP to Python, developing Python applications using Flask, and implementing machine learning techniques to optimize processes and enhance security measures."),
    ("Can you provide details about your previous experience in data analysis and machine learning?", "In my previous roles, I conducted data analysis on stock market data, optimized backend infrastructure, and developed machine learning models for various projects."),
    ("What are your key accomplishments in your current job role?", "Some of my key accomplishments in my current role include spearheading the migration of the code base from PHP to Python, developing robust Python applications, and implementing machine learning techniques for process optimization."),
    ("How do you stay updated with the latest trends and technologies in your field?", "I stay updated with the latest trends and technologies by regularly reading industry publications, attending workshops and conferences, and participating in online courses and webinars. Additionally, I am an active member of professional networking platforms where I engage with peers and experts in the field.")
]

# Preprocess QA pairs
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(tokens)

preprocessed_qa_pairs = [(preprocess(question), answer) for question, answer in qa_pairs]

# Split questions and answers
questions, answers = zip(*preprocessed_qa_pairs)

# Create Streamlit app
# Create Streamlit app
def main():
    # Set page config with logo and title
    logo_image = 'title.png'  # Provide the path to your logo image
    st.set_page_config(page_title="samir.ai", page_icon=logo_image)

    st.title("Samir.ai")
    st.markdown("Ask me anything regarding my profile!")

    user_input = st.text_input("You:")
    if st.button("Ask"):
        response = generate_response(user_input)
        st.text_area("Samir.ai:", value=response, height=100, max_chars=None, key=None)

# Define function to generate response
def generate_response(user_input):
    preprocessed_input = preprocess(user_input)
    max_similarity = -1
    best_response = "I'm sorry, I didn't understand your question."

    # Iterate through preprocessed questions and find the most similar one
    for idx, question in enumerate(questions):
        similarity = calculate_similarity(preprocessed_input, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_response = answers[idx]

    return best_response

# Define function to calculate similarity between two strings
def calculate_similarity(input_text, question):
    # Tokenize input text and question
    input_tokens = set(word_tokenize(input_text))
    question_tokens = set(word_tokenize(question))

    # Calculate Jaccard similarity
    intersection = len(input_tokens.intersection(question_tokens))
    union = len(input_tokens.union(question_tokens))
    similarity = intersection / union if union != 0 else 0

    return similarity

if __name__ == "__main__":
    main()
