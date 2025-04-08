from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Carregar modelo SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

def gerar_embedding(texto):
    return model.encode(texto, convert_to_tensor=True)

# 1. Carregar o modelo QA
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# 2. Carregar o dataset estruturado (escolha uma das opções abaixo)
 
# Opção A: Carregar de um arquivo JSON
try:
    # Se você salvou o dataset como JSON
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    # Converter para um formato mais fácil de usar
    qa_examples = []
    for item in dataset_json:  # Iterar diretamente sobre a lista
        context = item["context"]
        for qa in item["questions"]:
            question = qa["question"]
            answer = qa["answer"] if "answer" in qa else ""
            qa_examples.append({"context": context, "question": question, "answer": answer})
    
except Exception as e:
    print(f"Erro ao carregar JSON: {e}")
    qa_examples = []

# Opção B: Usar o dataset que você acabou de criar na memória

# Pré-processar as perguntas do dataset para gerar embeddings
for exemplo in qa_examples:
    exemplo["embedding"] = gerar_embedding(exemplo["question"])

# 3. Função para responder a perguntas usando o dataset estruturado
def responder_com_dataset(pergunta_usuario):
    """
    Responde à pergunta do usuário utilizando o dataset estruturado:
    1. Busca perguntas similares no dataset usando embeddings.
    2. Se encontrar uma pergunta similar, retorna a resposta associada.
    3. Se não, usa o contexto mais relevante para gerar uma resposta.
    """
    # Encontrar pergunta similar
    pergunta_similar, resposta = encontrar_pergunta_similar(pergunta_usuario)
    if pergunta_similar:
        return f"Resposta para '{pergunta_similar}': {resposta}"
    
    # Se não encontrou pergunta similar, usar o pipeline QA
    melhor_contexto = " ".join(set(ex["context"] for ex in qa_examples))
    try:
        resultado = qa_pipeline(question=pergunta_usuario, context=melhor_contexto)
        return resultado["answer"]
    except Exception as e:
        return f"Não consegui encontrar uma resposta: {str(e)}"

def encontrar_pergunta_similar(pergunta_usuario):
    """
    Encontra a pergunta mais similar no dataset usando similaridade de cosseno.
    """
    embedding_usuario = gerar_embedding(pergunta_usuario)
    similaridades = []
    
    for exemplo in qa_examples:
        similaridade = cosine_similarity(embedding_usuario.unsqueeze(0).numpy(), exemplo["embedding"].unsqueeze(0).numpy())
        similaridades.append((similaridade, exemplo))
    
    # Ordenar pela maior similaridade
    similaridades.sort(key=lambda x: x[0], reverse=True)
    
    # Retornar a pergunta mais similar e sua resposta, se a similaridade for alta o suficiente
    if similaridades[0][0] > 0.9:
        return similaridades[0][1]["question"], similaridades[0][1]["answer"]
    else:
        #fallback
        return None, "Desculpe, não entendi sua pergunta. Pode reformular?"

# 4. Loop principal de interação
if __name__ == "__main__":
    print("Chatbot sobre Alzheimer.\nDigite sua pergunta ou 'sair' para encerrar.")
    
    # Verificar se o dataset foi carregado corretamente
    if not qa_examples:
        print("AVISO: O dataset não foi carregado. Usando apenas o modelo QA.")
    else:
        print(f"Dataset carregado com {len(qa_examples)} exemplos de perguntas e respostas.")
    
    while True:
        try:
            pergunta = input("Você: ")
            if pergunta.lower() == 'sair':
                break
            
            if not pergunta.strip():
                print("Chatbot: Por favor, faça uma pergunta.")
                continue
                
            resposta = responder_com_dataset(pergunta)
            print("Chatbot:", resposta)
        except KeyboardInterrupt:
            print("\nEncerrando o programa...")
            break
        except Exception as e:
            print(f"Erro inesperado: {e}")