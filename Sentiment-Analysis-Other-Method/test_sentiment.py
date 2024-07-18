import re
import string
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def analyze_sentiment_hindi(sentence):
    """Analyzes the sentiment of a Hindi sentence.

    Args:
        sentence (str): The Hindi sentence to analyze.

    Returns:
        float: A sentiment score between -2 (very negative) and 2 (very positive).
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    sentiment_class = outputs.logits.argmax().item()
    return sentiment_class - 2

conversation_path = './conversation.txt' 
with open(conversation_path, encoding="utf8") as file:
    text = file.read()

hindi_punctuation = '|,!?;:'
all_punctuation = string.punctuation + hindi_punctuation
cleaned_text = text.translate(str.maketrans('', '', all_punctuation))

ra_sentences = []
borrower_sentences = []
overall_sentences = []
for line in cleaned_text.splitlines():
    if line.startswith("R"):
        ra_sentences.append(line[3:].strip())
        overall_sentences.append(line[3:].strip())
    elif line.startswith("B"):
        borrower_sentences.append(line[2:].strip())
        overall_sentences.append(line[2:].strip())
    

ra_sentiments = [analyze_sentiment_hindi(sentence) for sentence in ra_sentences]
borrower_sentiments = [analyze_sentiment_hindi(sentence) for sentence in borrower_sentences]
overall_sentiments = [analyze_sentiment_hindi(sentence) for sentence in overall_sentences]
overall = analyze_sentiment_hindi(cleaned_text)

print(f"RA:  {ra_sentiments}")
print(f"B: {borrower_sentiments}")
print(f"Overall: {overall_sentiments}")
print(f"OA: {overall}")

# Plotting Overall Sentiment
plt.figure(figsize=(10, 5))
plt.plot(overall_sentiments, marker='o', linestyle='-', label="Overall")
plt.title("Sentiment Analysis")
plt.xlabel("Sentence Number")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.legend()
plt.savefig('test_sentiment')

# Plotting Recovery Agent Sentiment
plt.figure(figsize=(10, 5))
plt.plot(ra_sentiments, marker='o', linestyle='-', label="Recovery Agent")
plt.title("Sentiment Analysis")
plt.xlabel("Sentence Number")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.legend()
plt.savefig('test_sentiment_ra')

# Plotting Borrower Sentiment
plt.figure(figsize=(10, 5))
plt.plot(borrower_sentiments, marker='o', linestyle='-', label="Borrower")
plt.title("Sentiment Analysis")
plt.xlabel("Sentence Number")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.legend()
plt.savefig('test_sentiment_b')