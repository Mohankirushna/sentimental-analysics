# Sentiment Analysis Chatbot with GUI

This project is a simple sentiment analysis chatbot that classifies user input text into **positive**, **negative**, or **neutral** sentiments using a combination of machine learning and VADER sentiment analysis. The chatbot has a graphical user interface (GUI) built with Tkinter for easy interaction.

---

## Features

- Sentiment classification using a Multinomial Naive Bayes model trained on the SST-2 dataset.
- VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer integration for fine-grained sentiment scoring.
- Keyword-based sentiment label adjustment for improved accuracy.
- Interactive chat interface built with Tkinter.
- Real-time chat display with user input and chatbot responses.
- Visualization of sentiment distribution in the training dataset via a bar graph.
- Save/load functionality for the trained model and vectorizer using `pickle`.

---

## How It Works

1. The project downloads and loads the SST-2 sentiment dataset.
2. The text data is vectorized using `TfidfVectorizer`.
3. A Multinomial Naive Bayes model is trained to classify sentences as positive or negative.
4. VADER sentiment analyzer scores the input text to provide compound sentiment scores.
5. Keyword matching further refines sentiment classification.
6. User inputs are taken via the GUI, processed, and responses are generated and displayed.
7. Users can view the distribution of sentiment labels in the dataset with a button click.

---

## Usage

1. Run the script:
    ```bash
    python your_script_name.py
    ```

2. Enter your text in the chat input box and press **Send** or hit **Enter**.

3. The chatbot will respond with the detected sentiment label.

4. Click **Show Sentiment Distribution Graph** to view the sentiment breakdown of the training data.

5. Type `exit` to quit the application.

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk
- tkinter (usually included with Python)
- Pillow (for image display in Tkinter)

---

## Installation

Install required Python packages with:

```bash
pip install pandas numpy scikit-learn matplotlib nltk Pillow
```
