import re
import spacy
def preprocess_text(text):
    if text is None:
        return ""
    # Parse the text using SpaCy
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    doc = nlp(text)
    
    # Lemmatize and remove stopwords and special characters
    preprocessed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and not token.is_digit]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(preprocessed_tokens)
    
    # Remove remaining special characters using regex
    preprocessed_text = re.sub(r'[^a-zA-Z\s]', '', preprocessed_text)    
    return preprocessed_text
def recommend_books_based_on_input(user_input, similarity_matrix, books_df, top_n=5):
    # Preprocess user input
    user_input = preprocess_text(user_input)

    # Find the indices of books that match the user input
    matching_indices = books_df.index[books_df['combined_text'].str.contains(user_input, case=False)].tolist()

    # Ensure that matching indices are within the bounds of the similarity matrix
    #matching_indices = [idx for idx in matching_indices if idx < len(similarity_matrix)]

    if not matching_indices:
        print("No matching books found for the input.")
        return None

    # Calculate the average similarity scores for matching books
    average_similarity_scores = []
    for idx in matching_indices:
        similarity_scores = similarity_matrix[idx]
        average_similarity_scores.append(sum(similarity_scores) / len(similarity_scores))

    # Sort matching books by average similarity scores
    sorted_indices = [x for _, x in sorted(zip(average_similarity_scores, matching_indices), reverse=True)]

    # Recommend top N similar books
    recommended_books = books_df.iloc[sorted_indices[:top_n]]
    return recommended_books[['title', 'authors']]