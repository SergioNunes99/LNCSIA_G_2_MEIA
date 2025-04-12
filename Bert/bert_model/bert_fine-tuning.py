from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load SBERT model
# This model is used to generate embeddings for text, which are numerical representations of the text's meaning.
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    """
    Generate an embedding for the given text using the SBERT model.
    Embeddings are used to calculate similarity between texts.
    """
    return model.encode(text, convert_to_tensor=True)

# 1. Load the QA model
# This model is fine-tuned on SQuAD and is used to answer questions based on a given context.
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# 2. Load the structured dataset
# Option A: Load the dataset from a JSON file
try:
    # Open and load the dataset from a JSON file
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    # Convert the dataset into a list of examples with context, question, and answer
    qa_examples = []
    for item in dataset_json:  # Iterate over each item in the dataset
        context = item["context"]
        for qa in item["questions"]:
            question = qa["question"]
            answer = qa["answer"] if "answer" in qa else ""  # Handle cases where the answer might be missing
            qa_examples.append({"context": context, "question": question, "answer": answer})
    
except Exception as e:
    # Handle errors if the dataset cannot be loaded
    print(f"Error loading JSON: {e}")
    qa_examples = []

# Option B: Use the dataset created in memory (if applicable)

# Preprocess the dataset by generating embeddings for each question
# This step prepares the dataset for similarity calculations.
for example in qa_examples:
    example["embedding"] = generate_embedding(example["question"])

# List of trivial or irrelevant questions that the chatbot should not answer
trivial_questions = [
    "what's your name", "who are you", "how are you", "why so serious", "thanks", "hello", "hi",
    "what time is it", "what day is it", "how old are you", "where are you from", "what's up",
    "can you help me", "are you a robot", "do you speak english", "tell me a joke", "how's the weather",
    "what do you do", "are you real", "what is your purpose", "do you like me", "can you hear me"
]

def check_trivial_question(user_question):
    return any(user_question.lower() in trivial for trivial in trivial_questions)

def is_question(user_input):
    return "?" in user_input.strip()

def select_best_context(user_question):
    """
    Select the most relevant context from the dataset based on cosine similarity
    between the user's question and the context embeddings.
    """
    user_embedding = generate_embedding(user_question)  # Generate embedding for the user's question
    best_context = None
    highest_similarity = 0

    for example in qa_examples:
        # Compute cosine similarity between the user's question and the context
        similarity = cosine_similarity(
            user_embedding.unsqueeze(0).numpy(),
            generate_embedding(example["context"]).unsqueeze(0).numpy()
        )
        # Update the best context if a higher similarity is found
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_context = example["context"]

    # Return the best context if the similarity exceeds the threshold
    return best_context if highest_similarity > 0.75 else None

def answer_with_dataset(user_question):
    """
    Answer the user's question using the structured dataset and the QA model.
    """
    # Check if the question is trivial
    if check_trivial_question(user_question):
        return "Sorry, I can't answer that question."

    # Find a similar question in the dataset
    similar_question, answer = find_similar_question(user_question)
    if similar_question:
        return f"{answer}"
    
    # Select the best context for the question
    best_context = select_best_context(user_question)
    if best_context:
        try:
            # Use the QA model to find the answer based on the best context
            result = qa_pipeline(question=user_question, context=best_context)
            return result["answer"]
        except Exception as e:
            # Handle errors during the QA process
            return f"I couldn't find an answer: {str(e)}"
    else:
        # Fallback if no relevant context is found
        return "Sorry, I couldn't find a relevant context for your question. Please rephrase the question."

def find_similar_question(user_question):
    """
    Find the most similar question in the dataset using cosine similarity.
    """
    user_embedding = generate_embedding(user_question)  # Generate embedding for the user's question
    similarities = []
    
    for example in qa_examples:
        # Compute cosine similarity between the user's question and dataset questions
        similarity = cosine_similarity(user_embedding.unsqueeze(0).numpy(), example["embedding"].unsqueeze(0).numpy())
        similarities.append((similarity, example))
    
    # Sort the similarities in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Return the most similar question and its answer if the similarity is high enough
    if similarities[0][0] > 0.75:  # Similarity threshold
        return similarities[0][1]["question"], similarities[0][1]["answer"]
    else:
        # Fallback if no similar question is found
        return None, "Sorry, I didn't understand your question. Can you rephrase it?"

# 4. Main interaction loop
if __name__ == "__main__":
    print("Alzheimer's Chatbot.\nType your question or 'exit' to quit.")
    
    # Check if the dataset was loaded correctly
    if not qa_examples:
        print("WARNING: The dataset was not loaded. Using only the QA model.")
    else:
        print(f"Dataset loaded with {len(qa_examples)} question and answer examples.")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: It was nice talking to you. Goodbye!")
                break
            
            if not user_input.strip():
                print("Chatbot: Please say something. I'm here to help!")
                continue
            
            # Check if the input is a question
            if is_question(user_input):
                answer = answer_with_dataset(user_input)
                print("Chatbot:", answer)
            else:
                # Respond with a friendly message if it's not a question
                print("Chatbot: That's interesting! Feel free to ask me anything.")
        except KeyboardInterrupt:
            # Handle user interruption (Ctrl+C)
            print("\nChatbot: Goodbye! Take care.")
            break
        except Exception as e:
            # Handle unexpected errors
            print(f"Unexpected error: {e}")