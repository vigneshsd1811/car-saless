import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import os.path

# Sample data from your code
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey there"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Goodbye", "See you later", "Bye"]
        }
    ]
}

# Required initialization
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]
lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Add words to the words list

        # Associate patterns with respective tags
        documents.append((word_list, intent['tag']))

        # Append tags to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Store the root words or lemmas
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Save the words list to a binary file
try:
    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    print("Words have been saved to 'words.pkl'.")
except Exception as e:
    print("An error occurred while saving the pickle file:", str(e))

# Load the words list from the pickle file
if os.path.exists('words.pkl'):
    try:
        with open('words.pkl', 'rb') as f:
            words = pickle.load(f)
        print("Words have been loaded from 'words.pkl'.")
    except Exception as e:
        print("An error occurred while loading the pickle file:", str(e))
else:
    print("'words.pkl' file does not exist.")

