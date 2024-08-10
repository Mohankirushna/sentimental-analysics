# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import urllib.request
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import Toplevel, scrolledtext
from PIL import Image, ImageTk

# Step 1: Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Step 2: Fetch and load the dataset
url = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'
urllib.request.urlretrieve(url, 'train.tsv')

data = pd.read_csv('train.tsv', sep='\t', header=None, names=['text', 'label'])

# Change labels from 0 and 1 to 'negative' and 'positive'
data['label'] = data['label'].map({0: 'negative', 1: 'positive'})

# Step 3: Preprocess the data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Step 4: Build the machine learning model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Step 5: Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze the sentiment of the given text using VADER."""
    sentiment = sia.polarity_scores(text)
    print(f"Sentiment Scores: {sentiment}")  # Debugging statement
    return sentiment

def get_sentiment_label(sentiment_scores, text):
    """Determine the sentiment label from sentiment scores and keywords."""
    compound = sentiment_scores.get('compound', None)  # Use .get() to avoid KeyError
    print(f"Compound Score: {compound}")  # Debugging statement
    
    if compound is None:
        print("Error: Compound score is missing.")
        return 'neutral'  # Default to neutral if the score is missing

    text_lower = text.lower()
    print(f"Text Lower: {text_lower}")  # Debugging statement
    
    # Define positive and negative keywords
    negative_keywords = [
        "abandoned", "abused", "angry", "annoyed", "anxious", "apathetic", "appalled", "ashamed",  
        "bad", "betrayed", "bitter", "bored", "broken", "confused", "crushed", "defeated", "depressed",  
        "disappointed", "disgusted", "distraught", "distressed", "embarrassed", "enraged", "excluded",  
        "forgot", "forgotten", "frustrated", "guilty", "hated", "heartbroken", "helpless", "hopeless",  
        "hurt", "ignored", "inadequate", "insecure", "isolated", "jealous", "left me", "left my friends",  
        "let down", "lonely", "lost", "miserable", "neglected", "offended", "overwhelmed", "rejected",  
        "regretful", "remorseful", "resentful", "sad", "scared", "shameful", "shocked", "sorrowful",  
        "stressed", "terrible", "unappreciated", "uncomfortable", "unhappy", "unloved", "upset", "useless",  
        "vulnerable", "weak", "worthless", "not happy", "is not joy", "don't love", "not excited", "great", 
        "not wonderful", "not the best day", "not fantastic", "not awesome", "not amazing", "not brilliant", 
        "not cheerful", "not delightful", "not elated", "not encouraged", "not enthusiastic", "not excellent", 
        "not fabulous", "not fantastic", "not flawless", "not fortunate", "not free", "not glad", "not gleeful", 
        "not good", "not happy", "not healthy", "not helpful", "not ideal", "not impeccable", "not incredible", 
        "not kind", "not lucky", "not marvelous", "not nice", "not optimistic", "not perfect", "not pleased", 
        "not positive", "not proud", "not radiant", "not satisfied", "not smiling", "not stunning", 
        "not superb", "not thankful", "not triumphant", "not victorious", "not winning", "not wonderful", 
        "not zealous"
    ]
    positive_keywords = [
        "happy", "joy", "love", "excited", "great", "wonderful", "best day", "fantastic", "awesome",
        "amazing", "brilliant", "cheerful", "delightful", "elated", "encouraged", "enthusiastic",
        "excellent", "fabulous", "fantastic", "flawless", "fortunate", "free", "glad", "gleeful",
        "good", "happy", "healthy", "helpful", "ideal", "impeccable", "incredible", "kind",
        "lucky", "marvelous", "nice", "optimistic", "perfect", "pleased", "positive", "proud",
        "radiant", "satisfied", "smiling", "stunning", "superb", "thankful", "triumphant",
        "victorious", "winning", "wonderful", "zealous"
    ]
    
    # Check for positive keywords first
    if any(keyword in text_lower for keyword in positive_keywords):
        print("Matched Positive Keyword")  # Debugging statement
        return 'positive'
    
    # Then check for negative keywords
    if any(keyword in text_lower for keyword in negative_keywords):
        print("Matched Negative Keyword")  # Debugging statement
        return 'negative'
    
    # If no keywords match, use compound score
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def chatbot_response(user_input):
    """Generate a response from the chatbot."""
    sentiment_scores = analyze_sentiment(user_input)
    sentiment_label = get_sentiment_label(sentiment_scores, user_input)
    
    response = f"I sense that your sentiment is {sentiment_label}."
    return response

def show_graph():
    """Open a new window to display the sentiment distribution graph."""
    # Create a new top-level window
    graph_window = Toplevel(root)
    graph_window.title("Sentiment Distribution")

    # Plot the graph in the new window
    plt.figure(figsize=(6, 4))
    data['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Sentiment Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Save the plot as an image
    plt.savefig('sentiment_distribution.png')
    plt.close()

    # Load and display the image in the Tkinter window
    img = Image.open('sentiment_distribution.png')
    img_tk = ImageTk.PhotoImage(img)
    label = tk.Label(graph_window, image=img_tk)
    label.image = img_tk  # Keep a reference to avoid garbage collection
    label.pack()

def on_send(event=None):
    """Handle the sending of user input and display chatbot response."""
    user_input = entry.get("1.0", tk.END).strip()
    if user_input.lower() == 'exit':
        root.quit()
    else:
        response = chatbot_response(user_input)
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"You: {user_input}\n")
        chat_window.insert(tk.END, f"Chatbot: {response}\n\n")
        chat_window.config(state=tk.DISABLED)
        entry.delete("1.0", tk.END)

# Create the main application window
root = tk.Tk()
root.title("Sentiment Analysis Chatbot")

# Create a text widget for the chat window
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=15, width=60)
chat_window.pack(padx=10, pady=10)

# Create an entry widget for user input
entry = tk.Text(root, height=4, width=60)
entry.pack(padx=10, pady=5)
entry.bind("<Return>", on_send)  # Bind the Enter key to the on_send function

# Create a send button
send_button = tk.Button(root, text="Send", command=on_send)
send_button.pack(pady=5)

# Create a button to show the sentiment distribution graph
graph_button = tk.Button(root, text="Show Sentiment Distribution Graph", command=show_graph)
graph_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
