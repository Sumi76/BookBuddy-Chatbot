from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from recommendation import recommend_books_based_on_input
import joblib
import numpy as np
import pandas as pd

#loading necessary data
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
similarity_matrix = np.load('similarity_matrix.npy')
books_df = pd.read_csv('books_data.csv')

app = FastAPI()

class DialogflowRequest(BaseModel):
    queryResult: dict

class BookRecommendation(BaseModel):
    recommendation: str

@app.post("/")
async def handle_dialogflow_request(dialogflow_request: DialogflowRequest):
    intent_name = dialogflow_request.queryResult.get("intent", {}).get("displayName")
    if intent_name == "author-name - yes":
        parameters = dialogflow_request.queryResult.get("parameters", {})
        person_name = parameters.get("person", {}).get("name")
        recommendation = recommend_books_based_on_input(user_input=person_name ,similarity_matrix=similarity_matrix,books_df= books_df, top_n=5)
        return {"fulfillmentText": f"Here are the top 5 book recommendations for {person_name}:\n{format_recommendations(recommendation)}\n"}
    
    elif intent_name == "author-name - no - custom":
        parameters = dialogflow_request.queryResult.get("parameters", {})
        book_genre = parameters.get("generes")  # Corrected: "generes" instead of "genre"
        recommendation = recommend_books_based_on_input(user_input=book_genre, similarity_matrix=similarity_matrix,books_df= books_df, top_n=5)
        return {"fulfillmentText": f"Here are the top 5 book recommendations for {book_genre}:\n{format_recommendations(recommendation)}\n"}
    else:
        return {"fulfillmentText": "Sorry, I couldn't understand your request."}
    
def format_recommendations(recommendation):
    # Format recommendations as a string to be displayed
    if not recommendation.empty:
        formatted_recommendations = "\n\n".join([f"[\n{book['title']} by {book['authors']}\n]" for _, book in recommendation.iterrows()])
    else:
        formatted_recommendations = "No recommendations found."
    return formatted_recommendations












