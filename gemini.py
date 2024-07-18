from langchain_google_genai import GoogleGenerativeAI

def summary(text):
    training_data = text
    prompt = """Analyze the provided Hindi conversation between a recovery agent and a borrower regarding a default payment. Your analysis should include:
                1. A concise summary of the conversation (in English) (not more than 100 words)
            """

    llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="YOUR_API_KEY")
    
    response = llm.invoke(prompt+"\n"+training_data)

    return response

def next_steps(text):
    training_data = text
    prompt = """Analyze the provided Hindi conversation between a recovery agent and a borrower regarding a default payment. Your analysis should include:
                1. Key actions or next steps identified from the conversation
                   For key actions, list 3-5 bullet points.
            """

    llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="YOUR_API_KEY")

    response = llm.invoke(prompt+"\n"+training_data)

    return response

def sentiment_g(text):
    training_data = text
    prompt = """Analyze the provided Hindi conversation between a recovery agent and a borrower regarding a default payment. Your analysis should include:
                1. Sentiment analysis of both the recovery agent and the borrower
                In the sentiment analysis, discuss the overall tone of each participant and how it changes throughout the conversation (Initial, Mid, End). Use specific examples from the text to support your analysis (convert the example text to English if necessary). 
            """

    llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="YOUR_API_KEY")

    response = llm.invoke(prompt+"\n"+training_data)

    return response