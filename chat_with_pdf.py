# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 03:03:59 2024

@author: sayan
"""

import PyPDF2
import os
import warnings

from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def take_User_pdf():

    """ 
    This function takes file path of the pdf, then verifies:
    1. If the file exists and can be accessed by the code or not.
    2. If the file exists, then whether it is a pdf file or not.
    3. Whether pdf file is less than 5 Mb or not, this is a constraint.

    In case any of the above consitions are failed to be fulfilled, it prompts the user to try again with the valid file path.

    The function also returns the pdf file path and the file path where the mp3 file is to be saved.
    By default, the mp3 file will have the same name as the pdf file and will be saved in the same location.

    The function returns the file path of the pdf as pdf_path, and the file path of the mp3 file as svae_directory.
    """
    #One can call another api here to take file_path, no need to take user input.
    pdf_path = input("Please enter the file path of the pdf: ")      
    file_name = os.path.basename(pdf_path)
    file_name, file_extension = os.path.splitext(file_name)
    
    #Checks if the file exists and can be accessed by the code or not.
    try:                                                                   
        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)        
    except Exception as error:
        print("Error: {}\n Please try again".format(error))
        #recursive calling
        pdf_path, save_directory=take_user_pdf()
    #Checks if the file exists, then whether it is a pdf file or not.
    if file_extension!=".pdf":
        print("Only pdf formats are supported, please try again.")
        #recursive calling
        pdf_path, save_directory=take_user_pdf()
    #Whether pdf file is less than 5 Mb or not, this is a constraint.
    if pdf_size_mb > 5:
        print("PDF file size exceeds the maximum limit of 5 MB.\n PLease try again.")
        #recursive calling
        pdf_path, save_directory=take_user_pdf()

    return pdf_path


def extract_text_from_pdf(path: str):

    """
    Extract text from specified pages of a PDF file or from all pages if no pages are specified.

    Parameters:
    path (str): The path to the PDF file.

    Returns:
    list: The extracted text from all pages in the pdf as a list, page-wise.

    Description:
    This function extracts text from all pages if no pages are specified. 
    The function returns the extracted text as a list of strings, ordeed page-wise.

    Notes:
    - The function uses the PyPDF2 library to read the PDF file. Make sure PyPDF2 is installed.
    """

    #opens the pdf file from path
    text = ""
    file=open(path, "rb")  
    reader = PyPDF2.PdfReader(file)
    pages=[]
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        pages.append(text)
    
    return pages


def vector_database(pages_list:list, chunk_size=1000, chunk_overlap=0, model="all-MiniLM-L6-v2"):

    """
    Create a vector database from a list of pages using specified chunking and embedding parameters.

    Parameters:
    pages_list (list): A list of strings, where each string represents the text of a page.
    chunk_size (int, optional): The maximum size of each chunk. Default is 1000 characters.
    chunk_overlap (int, optional): The number of characters to overlap between chunks. Default is 0.
    model (str, optional): The name of the model to use for creating embeddings. Default is "all-MiniLM-L6-v2".

    Returns:
    Chroma: A Chroma vector database containing the embedded chunks.

    Description:
    This function takes a list of pages, splits the text of each page into smaller chunks,
    creates embeddings for each chunk using the specified model, and stores the embeddings
    in a Chroma vector database.
    """
    
    documents = [Document(page) for page in pages_list]

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    
    #for embedding the text into a vector database
    embedding_function = SentenceTransformerEmbeddings(model_name=model)
    db = Chroma.from_documents(split_docs, embedding_function)
    return db

def search_vc_database(vc_database, query: str):
    """
    Search a vector database for relevant documents based on a query.

    Parameters:
    vc_database (Chroma): The Chroma vector database to search.
    query (str): The query string to search for in the database.

    Returns:
    list: A list of relevant documents matching the query.

    Description:
    This function takes a Chroma vector database and a query string as input.
    It converts the database into a retriever object and performs a similarity
    search based on the query. The function returns a list of documents that are
    relevant to the query.
    """
    docsearch = vc_database.as_retriever(search_kwargs={"k": 1})
    docs = docsearch.invoke(query)
    return docs

def reply_queries(search_res:list, query: str, chain):
    """
    Process a query using a question-answering chain and print the response.

    Parameters:
    search_res (list): A list of documents that are relevant to the query.
    query (str): The query string to be processed.
    chain: The question-answering chain object used to generate responses.

    Returns:
    None

    Description:
    This function takes a list of relevant documents, a query string, and a question-answering
    chain object. It runs the chain with the input documents and the query, then prints the response.
    """
    output = chain.run(input_documents=search_res, question=query)
    print(r"response: ", output)
    

def accept_query():

    """
    Prompt the user to enter a query and return the input.

    Parameters:
    None

    Returns:
    str: The user's input query.

    Description:
    This function prompts the user to enter a query. It also provides instructions
    to enter 'STOP' if the user wants to end the Q&A session. The user's input is
    then returned as a string.
    """
    
    print(r"If you want to stop the qna session, please enter: 'STOP'")
    query = input(r"Please enter your query: ")
    return query

def main(api_key):
    
    """
    Main function to process a PDF file and handle a Q&A session using OpenAI's GPT-3.5-turbo.

    Parameters:
    api_key (str): The API key for accessing the OpenAI service.

    Returns:
    None

    Description:
    This function performs the following steps:
    1. Prompts the user to provide the file path of a PDF.
    2. Processes the PDF to extract text from its pages.
    3. Splits the extracted text into chunks and embeds them into a vector database.
    4. Initializes a question-answering chain using OpenAI's GPT-3.5-turbo.
    5. Enters a loop to accept user queries, search the vector database for relevant documents,
       and provide answers using the question-answering chain.
    6. Allows the user to exit the Q&A session by entering 'STOP'.

    The function ensures that the PDF processing, vector database creation, and query handling
    are performed in an interactive session with the user.
    """
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    pdf_path=take_User_pdf()
    print("Please wait, processing pdf...")
    ext_pages=extract_text_from_pdf(pdf_path)
    
    vc_db=vector_database(pages_list=ext_pages, chunk_size=1000, chunk_overlap=0)
    print("Pdf processed.")
    #connect with open ai model, initiale qna chain
    chain = load_qa_chain(ChatOpenAI(openai_api_key=api_key,temperature=0,model="gpt-3.5-turbo"), chain_type="stuff")

    while True:
        
        query=accept_query()
        if query=='STOP':
            print("Exiting Session")
            break
        print("....")
        relevant_docs=search_vc_database(vc_db, query=query)
        reply_queries(search_res=relevant_docs, query=query, chain=chain)
        print("....")

if __name__ == '__main__':

    api_key="Enter your api key here"
    main(api_key)