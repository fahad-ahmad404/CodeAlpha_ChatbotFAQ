# ==============================================================
# CodeAlpha Artificial Intelligence Internship
# Project      : Chatbot for FAQs
# Developed By : Fahad Ahmad
# Student ID   : CA/DF1/189817
# Description  : A console-based FAQ chatbot. It preprocesses a set of
#                stored question-answer pairs using NLTK (tokenizing,
#                cleaning, lemmatizing), matches the user's question to
#                the most similar stored question using TF-IDF vectors
#                and cosine similarity, and returns the best-matching
#                answer.
# ==============================================================

from __future__ import annotations

import json                                          # Used to load the FAQ data from faqs.json
import string                                        # Provides the list of punctuation characters to strip

import nltk                                          # NLP toolkit used for tokenizing and lemmatizing text
from nltk.corpus import stopwords                    # Common words (the, is, at...) to filter out
from nltk.stem import WordNetLemmatizer               # Reduces words to their base/dictionary form
from sklearn.feature_extraction.text import TfidfVectorizer   # Converts text into numeric vectors
from sklearn.metrics.pairwise import cosine_similarity          # Measures similarity between vectors

FAQ_FILE = "faqs.json"                               # File containing the stored question-answer pairs
SIMILARITY_THRESHOLD = 0.3                           # Minimum similarity score to accept a match


def ensure_nltk_data() -> None:
    """Download required NLTK data files if they aren't already present."""
    required_packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for package in required_packages:
        try:
            nltk.data.find(f"tokenizers/{package}" if "punkt" in package else f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)        # Fetch the missing NLTK resource silently


def load_faqs(file_path: str) -> list[dict]:
    """Load the FAQ question-answer pairs from a JSON file."""
    with open(file_path, mode="r", encoding="utf-8") as faq_file:
        return json.load(faq_file)                    # Parse the JSON file into a list of dicts


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Clean and normalize text for comparison.

    Steps: lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize.

    Args:
        text: raw input text.
        lemmatizer: an initialized WordNetLemmatizer.
        stop_words: a set of stopwords to exclude.

    Returns:
        The cleaned, lemmatized text as a single string.
    """
    text = text.lower()                                          # Normalize case so "Order" == "order"
    text = text.translate(str.maketrans("", "", string.punctuation))  # Strip out punctuation marks
    tokens = nltk.word_tokenize(text)                             # Split the text into individual words
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]                                                              # Remove stopwords and lemmatize what's left
    return " ".join(cleaned_tokens)                               # Rejoin the tokens into a single string


def build_vectorizer(questions: list[str]) -> tuple[TfidfVectorizer, "scipy.sparse.spmatrix"]:
    """Fit a TF-IDF vectorizer on the preprocessed FAQ questions."""
    vectorizer = TfidfVectorizer()                                # Converts text into TF-IDF weighted vectors
    question_vectors = vectorizer.fit_transform(questions)        # Learn vocabulary and vectorize the questions
    return vectorizer, question_vectors


def find_best_match(
    user_question: str,
    vectorizer: TfidfVectorizer,
    question_vectors,
    faqs: list[dict],
    lemmatizer: WordNetLemmatizer,
    stop_words: set,
) -> tuple[str, float]:
    """
    Find the FAQ entry whose question is most similar to the user's question.

    Returns:
        A tuple of (best answer or fallback message, similarity score).
    """
    cleaned_input = preprocess_text(user_question, lemmatizer, stop_words)  # Clean the user's question
    input_vector = vectorizer.transform([cleaned_input])                   # Convert it into a TF-IDF vector

    similarities = cosine_similarity(input_vector, question_vectors)[0]    # Compare against all FAQ questions
    best_index = similarities.argmax()                                     # Index of the most similar question
    best_score = similarities[best_index]                                  # How similar that match actually is

    if best_score < SIMILARITY_THRESHOLD:                                  # Reject weak/unrelated matches
        return "I'm sorry, I don't have an answer for that. Please contact support@example.com.", best_score

    return faqs[best_index]["answer"], best_score                          # Return the matching stored answer


def main() -> None:
    print("=" * 50)
    print("       FAQ CHATBOT")
    print("=" * 50)
    print("Ask me anything about orders, shipping, returns, or payments.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    ensure_nltk_data()                                    # Make sure required NLTK resources are available
    lemmatizer = WordNetLemmatizer()                      # Set up the lemmatizer once for reuse
    stop_words = set(stopwords.words("english"))          # Load the English stopword list once

    faqs = load_faqs(FAQ_FILE)                            # Load all stored FAQ question-answer pairs
    raw_questions = [faq["question"] for faq in faqs]     # Pull out just the questions for vectorizing
    cleaned_questions = [
        preprocess_text(question, lemmatizer, stop_words) for question in raw_questions
    ]                                                      # Preprocess every stored FAQ question up front
    vectorizer, question_vectors = build_vectorizer(cleaned_questions)  # Build the TF-IDF model once

    while True:
        user_input = input("You: ").strip()               # Read the user's question

        if user_input.lower() in ("exit", "quit"):         # Let the user end the chat
            print("Bot: Goodbye! Have a great day.")
            break

        if not user_input:
            print("Bot: Please type a question.")
            continue

        answer, score = find_best_match(
            user_input, vectorizer, question_vectors, faqs, lemmatizer, stop_words
        )                                                   # Find the closest matching FAQ answer
        print(f"Bot: {answer}")                             # Display the chatbot's response


if __name__ == "__main__":
    main()   # Run the program only when executed directly (not when imported)
