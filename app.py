import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available in the context, say "answer is not available in the context."
    
    Context:
    {context}?
    
    Question:
    {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def summarize_text(text):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = f"Summarize the following text:\n{text[:4000]}"  # Limit to 4000 characters
    response = model.predict(prompt)
    return response

def translate_text(text, target_language):
    translated_text = GoogleTranslator(source="auto", target=target_language).translate(text)
    return translated_text

def user_input(user_question, selected_language_code):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    output_text = response["output_text"]
    
    # Translate response if needed
    if selected_language_code != "en":  # Compare with "en" instead of "English"
        output_text = translate_text(output_text, selected_language_code)
    
    st.write("Reply:", output_text)

def main():
    st.set_page_config(page_title="Supreet's PDFChatbot")  
    st.header("Supreet's PDFChatbot 💁")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    language_options = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh-cn",
        "Hindi": "hi",
        "Telugu":"te"
    }
    selected_language_name = st.selectbox("Select Response Language", list(language_options.keys()))
    selected_language_code = language_options[selected_language_name]  

    if user_question:
        user_input(user_question, selected_language_code)  

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        
        if st.button("Summarize PDFs Individually"):
            with st.spinner("Summarizing each PDF..."):
                summaries = {}
                for pdf in pdf_docs:
                    pdf_name = pdf.name  # Get file name
                    raw_text = get_pdf_text([pdf])  # Process one PDF at a time
                    summary = summarize_text(raw_text)  # Summarize individually
                    summaries[pdf_name] = summary  # Store summary with filename
                
                st.subheader("PDF Summaries:")
                for pdf_name, summary in summaries.items():
                    st.write(f"**📄 {pdf_name}**")  
                    st.write(summary)
                    st.download_button(f"⬇ Download {pdf_name} Summary", summary, file_name=f"{pdf_name}_summary.txt")  
                    st.write("---")  

if __name__ == "__main__":
    main()
