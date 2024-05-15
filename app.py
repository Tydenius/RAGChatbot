import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

st.title("Converse com a Web 🌐")
st.caption("Este app permite que você converse com uma página da Web através do uso local do Llama-3 com técnica RAG")

# Get the webpage URL from the user
webpage_url = st.text_input("Digite a página da Web aqui", type="default")

#Carregar data da Web
if webpage_url:
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

# Criar Embedding do Ollama e o vetor de armazenamento
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Definir função do Ollama
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Configurar RAG
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

st.success(f"Loaded {webpage_url} successfully!")

# Implementar Chat
# Criar pergunta para a página
prompt = st.text_input("Pergunte algo sobre a página da Web")

# Converse com a página
if prompt:
    result = rag_chain(prompt)
    st.write(result)