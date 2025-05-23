import os
import torch
# from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACE_USER_ACCESS_TOKEN")

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings
    # Set up the environment variable for HuggingFace and initialize the desired model.
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

    # repo name for the model
    model_id = "tiiuae/falcon-7b-instruct"
    # load the model into the HuggingFaceHub
    llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 600, "max_length": 600})

    # Remove the token from environment variables so that embeddings initialization
    # does not pass the token to SentenceTransformer.
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)

    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": DEVICE}
    )


# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)


    # --> Build the QA chain, which utilizes the LLM and retriever for answering questions. 
    # By default, the vectorstore retriever uses similarity search. 
    # If the underlying vectorstore support maximum marginal relevance search, you can specify that as the search type (search_type="mmr").
    # You can also specify search kwargs like k to use when doing retrieval. k represent how many search results send to llm
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key = "question"
     #   chain_type_kwargs={"prompt": prompt} # if you are using prompt template, you need to uncomment this part
    )


# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    # Query the model
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    
    # Update the chat history
    chat_history.append((prompt, answer))
    
    # Return the model's response
    return answer

# Initialize the language model
init_llm()




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
    
#     # Load the tokenizer with use_fast=False to avoid the missing Qwen2Tokenizer error.
#     tokenizer = AutoTokenizer.from_pretrained(
#         "Qwen/Qwen2-7B-Instruct", 
#         trust_remote_code=True, 
#         use_fast=False
#     )
    
#     # Load the model with trust_remote_code=True.
#     model = AutoModelForCausalLM.from_pretrained(
#         "Qwen/Qwen2-7B-Instruct", 
#         trust_remote_code=True, 
#         device_map="auto"
#     )
    
#     # Create a text-generation pipeline using the loaded model and tokenizer.
#     pipe = pipeline(
#         "text-generation", 
#         model=model, 
#         tokenizer=tokenizer, 
#         max_new_tokens=600, 
#         temperature=0.1,
#         trust_remote_code=True
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
