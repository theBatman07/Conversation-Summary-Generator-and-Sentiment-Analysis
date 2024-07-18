import string
import matplotlib.pyplot as plt
from textblob import TextBlob
from googletrans import Translator


translator = Translator()

def analyze_sentiment_english(sentence):
    try:
        translation = translator.translate(sentence, dest='en').text
        analysis = TextBlob(translation)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Translation/Analysis Error: {e}")
        return 0

conversation_path = 'conversation.txt'
with open(conversation_path, encoding="utf8") as file:
    text = file.read()

hindi_punctuation = '|,!?;:'
all_punctuation = string.punctuation + hindi_punctuation
cleaned_text = text.translate(str.maketrans('', '', all_punctuation))


ra_sentences = []
borrower_sentences = []
overall_sentences = []
ra = ""
borrower = ""
for line in cleaned_text.splitlines():
    if line.startswith("R"):
        ra_sentences.append(line[3:].strip())
        overall_sentences.append(line[3:].strip())
        ra += line[3:].strip()
    elif line.startswith("B"):
        borrower_sentences.append(line[2:].strip())
        overall_sentences.append(line[2:].strip())
        borrower += line[2:].strip()

def sentiment_analyze():
    ra_sentiments = [analyze_sentiment_english(sentence) for sentence in ra_sentences]
    borrower_sentiments = [analyze_sentiment_english(sentence) for sentence in borrower_sentences]
    overall_sentiments = [analyze_sentiment_english(sentence) for sentence in ra_sentences + borrower_sentences]
    overall = analyze_sentiment_english(cleaned_text)
    ra_overall = analyze_sentiment_english(ra)
    borrower_overall = analyze_sentiment_english(borrower)

    print(f"Recovery Agent Sentiments throughout the conversation:  {ra_sentiments}\n")
    print(f"Borrower Sentiment throughout the conversation: {borrower_sentiments}\n")
    print(f"Recovery Agent Sentiment Overall:  {ra_overall}\n")
    print(f"Borrower Sentiment Overall: {borrower_overall}\n")
    print(f"Overall Sentiment thourghout the conversation: {overall_sentiments}\n")
    print(f"Overall Sentiment of the Conversation: {overall}\n")

    if ra_overall > 0:
        print("Recovery Agent had POSITIVE SENTIMENT\n")
    elif ra_overall < 0:
        print("Recovery Agent had NEGATIVE SENTIMENT\n")
    else:
        print("Recovery Agent had NEUTRAL SENTIMENT\n")

    if borrower_overall > 0:
        print("Borrower had POSITIVE SENTIMENT\n")
    elif borrower_overall < 0:
        print("Borrower had NEGATIVE SENTIMENT\n")
    else:
        print("Borrower had NEUTRAL SENTIMENT\n")

    # Plotting Overall Sentiment
    plt.figure(figsize=(10, 5))
    plt.plot(overall_sentiments, marker='o', linestyle='-', label="Overall")
    plt.title("Sentiment Analysis Overall")
    plt.xlabel("Sentence Number")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.legend()
    plt.savefig('Sentiment Analysis')

    # Plotting Recovery Agent Sentiment
    plt.figure(figsize=(10, 5))
    plt.plot(ra_sentiments, marker='o', linestyle='-', label="Recovery Agent")
    plt.title("Sentiment Analysis Recovery Agent")
    plt.xlabel("Sentence Number")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.legend()
    plt.savefig('Sentiment Analysis Recovery Agent')

    # Plotting Borrower Sentiment
    plt.figure(figsize=(10, 5))
    plt.plot(borrower_sentiments, marker='o', linestyle='-', label="Borrower")
    plt.title("Sentiment Analysis Borrower")
    plt.xlabel("Sentence Number")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.legend()
    plt.savefig('Sentiment Analysis Borrower')