from gemini import summary, next_steps, sentiment_g
from sentiment_analysis import sentiment_analyze


conversation_path = 'conversation.txt'
with open(conversation_path, encoding="utf8") as file:
    text = file.read()

print("SUMMARY")
print("-"*7)
print(summary(text=text))
print("KEY ACTIONS/ NEXT STEPS")
print("-"*23)
print(next_steps(text=text))
print("SENTIMENT ANALYSIS")
print("-"*18)
print(sentiment_g(text=text))
print("\nSENTIMENT ANALYSIS VALUES")
print("-"*23)
sentiment_analyze()