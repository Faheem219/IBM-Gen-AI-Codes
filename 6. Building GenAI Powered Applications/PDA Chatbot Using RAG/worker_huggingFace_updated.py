### IBM UPDATED WORKER SCRIPT ###
# import os
# import torch
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings  # use the standard embeddings class
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from dotenv import load_dotenv

# load_dotenv()
# huggingface_api_token = os.getenv("HUGGINGFACE_USER_ACCESS_TOKEN")

# # Check for GPU availability and set the appropriate device.
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Global variables
# conversation_retrieval_chain = None
# chat_history = []
# llm_hub = None
# embeddings = None

# # Function to initialize the language model and its embeddings
# def init_llm():
#     global llm_hub, embeddings
#     # Set the environment variable for HuggingFaceHub API token
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

#     # Specify the repo for the LLM model
#     model_id = "stabilityai/stablelm-tuned-alpha-7b"
#     llm_hub = HuggingFaceHub(
#         repo_id=model_id,
#         model_kwargs={
#             "temperature": 0.1,
#             "max_new_tokens": 600,
#             "max_length": 600
#         }
#     )

#     # Remove the token from the environment for security reasons
#     os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)

#     # Initialize embeddings using HuggingFaceEmbeddings from LangChain.
#     # This provides similar functionality to the instruct-based embeddings
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": DEVICE}
#     )

# # Function to process a PDF document
# def process_document(document_path):
#     global conversation_retrieval_chain

#     # Load the document using PyPDFLoader
#     loader = PyPDFLoader(document_path)
#     documents = loader.load()

#     # Split the document into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#     texts = text_splitter.split_documents(documents)

#     # Create a vectorstore from the split text chunks using Chroma
#     db = Chroma.from_documents(texts, embedding=embeddings)

#     # Build the RetrievalQA chain using the vectorstore retriever.
#     conversation_retrieval_chain = RetrievalQA.from_chain_type(
#         llm=llm_hub,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
#         return_source_documents=False,
#         input_key="question"
#     )

# # Function to process a user prompt
# def process_prompt(prompt):
#     global conversation_retrieval_chain, chat_history

#     # Query the RetrievalQA chain with the prompt and chat history.
#     output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
#     answer = output["result"]

#     # Update the chat history with the prompt and response.
#     chat_history.append((prompt, answer))
#     return answer

# # Initialize the language model and embeddings
# init_llm()





### CHATGPT WORKER SCRIPT (LOCAL DOWNLOAD) ###
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv

# load_dotenv()
# huggingface_api_token = os.getenv("HUGGINGFACE_USER_ACCESS_TOKEN")

# # Set device based on GPU availability.
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Global variables
# conversation_retrieval_chain = None
# chat_history = []
# llm_hub = None
# embeddings = None

# # Function to initialize the language model and its embeddings using direct model loading
# def init_llm():
#     global llm_hub, embeddings
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
#     # Load the tokenizer with use_fast=False to avoid the missing Qwen2Tokenizer error.
#     tokenizer = AutoTokenizer.from_pretrained(
#         "meta-llama/Llama-2-7b-chat-hf", 
#         use_auth_token=True
#     )
    
#     # Load the model with trust_remote_code=True.
#     model = AutoModelForCausalLM.from_pretrained(
#         "meta-llama/Llama-2-7b-chat-hf",
#         use_auth_token=True 
#     )
    
#     # Create a text-generation pipeline using the loaded model and tokenizer.
#     pipe = pipeline(
#         "text-generation", 
#         model=model, 
#         tokenizer=tokenizer, 
#         max_new_tokens=600, 
#         temperature=0.1,
#     )
    
#     # Wrap the pipeline with LangChain's HuggingFacePipeline interface.
#     llm_hub = HuggingFacePipeline(pipeline=pipe)
    
#     # Initialize embeddings using a pre-trained model.
#     embeddings = HuggingFaceInstructEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2", 
#         model_kwargs={"device": DEVICE}
#     )

# # Function to process a PDF document and build the retrieval QA chain.
# def process_document(document_path):
#     global conversation_retrieval_chain

#     # Load the document.
#     loader = PyPDFLoader(document_path)
#     documents = loader.load()
    
#     # Split the document into chunks.
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#     texts = text_splitter.split_documents(documents)
    
#     # Create an embeddings database using Chroma.
#     db = Chroma.from_documents(texts, embedding=embeddings)
    
#     # Build the RetrievalQA chain with a retriever using maximum marginal relevance.
#     conversation_retrieval_chain = RetrievalQA.from_chain_type(
#         llm=llm_hub,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
#         return_source_documents=False,
#         input_key="question"
#     )

# # Function to process a user prompt.
# def process_prompt(prompt):
#     global conversation_retrieval_chain, chat_history
    
#     # Query the model via the retrieval QA chain.
#     output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
#     answer = output["result"]
    
#     # Update chat history.
#     chat_history.append((prompt, answer))
    
#     return answer

# # Initialize the language model and embeddings.
# init_llm()





### CHATGPT WORKER SCRIPT (HUGGINGFACE INFERENCE) ###
import os
import torch
from typing import Optional, List
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings  # standard embeddings class
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydantic import Field

load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACE_USER_ACCESS_TOKEN")

# Check for GPU availability and set the appropriate device.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Custom LLM class that uses Hugging Face's InferenceClient with the Together provider.
class TogetherInferenceLLM(LLM):
    model_name: str = Field(...)
    max_tokens: int = Field(default=500)
    temperature: float = Field(default=0.1)
    provider: str = Field(...)
    api_key: str = Field(...)
    # Insert the system prompt here:
    system_prompt: str = Field(
        default="You are an AI assistant for an EdTech platform. Your primary tasks include generating personalized study plans based on quiz results, tracking sustainable study habits, and providing real-time quiz analysis. Your responses should be insightful, supportive, and actionable, guiding students effectively."
    )
    client: InferenceClient = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(provider=self.provider, api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "together_inference_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Prepare a message list in chat format.
        # Include the system prompt as a message before the user's message.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        msg = completion.choices[0].message
        if isinstance(msg, dict):
            return msg.get("content", "")
        else:
            return msg

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt
        }

# Function to initialize the language model and its embeddings.
def init_llm():
    global llm_hub, embeddings
    # Set the token as an environment variable for convenience.
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

    # Initialize our custom LLM using the Together InferenceClient.
    llm_hub = TogetherInferenceLLM(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        provider="together",
        api_key=huggingface_api_token,
        max_tokens=600,
        temperature=0.1
    )

    # Remove the token from the environment for security.
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)

    # Initialize embeddings using HuggingFaceEmbeddings.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

# Function to process a PDF document.
def process_document(document_path):
    global conversation_retrieval_chain

    # Load the document.
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # Split the document into chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    # Create a vector store from the document chunks using Chroma.
    db = Chroma.from_documents(texts, embedding=embeddings)

    # Build the RetrievalQA chain using the vector store's retriever.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )

# Function to process a user prompt.
def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history

    # Query the RetrievalQA chain with the prompt and chat history.
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]

    # Update the chat history.
    chat_history.append((prompt, answer))
    return answer

# Initialize the language model and embeddings.
init_llm()
