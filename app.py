import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ''
    if pdf_docs is not None:
        if isinstance(pdf_docs, list):
            for pdf in pdf_docs:
                pdf_reader = PdfReader(BytesIO(pdf.read()))
                for page in pdf_reader.pages:
                    text += page.extract_text()
        else:  # If pdf_docs is a single file
            pdf_reader = PdfReader(BytesIO(pdf_docs.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss-index')

def get_conversational_chain():
    prompt_template = '''
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provied context just say, 'answer is not available in the context', don't provide the wrong answser\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.1)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    try:
        new_db = FAISS.load_local('faiss-index', embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading vector store: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {'input_documents': docs, 'question': user_question}
    )

    # print(response)
    st.write('Reply: ', response['output_text'])


def menu():
    st.set_page_config('Chat with multiple PDF')
    st.header('Chat with Multiple PDF using Gemini')

    user_question = st.text_input('Ask a question from the PDF files')
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title('Menu: ')
        pdf_docs = st.file_uploader('Upload your PDF files and Click on the Submit & Process Button ')
        if st.button('Submit & Process'):
            with st.spinner("Procesing....."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success('Done')


if __name__ == "__main__":
    menu()