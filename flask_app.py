from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import tempfile

app = Flask(__name__)

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
    prompt = f"Summarize the following text:\n{text[:4000]}"
    response = model.predict(prompt)
    return response

def translate_text(text, target_language):
    translated_text = GoogleTranslator(source="auto", target=target_language).translate(text)
    return translated_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'pdf_files' not in request.files:
        return jsonify({'error': 'No PDF files uploaded'}), 400
    
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files:
        return jsonify({'error': 'No PDF files selected'}), 400
    
    # Save files temporarily
    temp_files = []
    for pdf in pdf_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.save(temp_file.name)
        temp_files.append(temp_file.name)
    
    try:
        raw_text = get_pdf_text(temp_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
            
        return jsonify({'message': 'PDFs processed successfully'}), 200
    except Exception as e:
        # Clean up temporary files in case of error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    language = data.get('language', 'en')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        output_text = response["output_text"]
        
        if language != "en":
            output_text = translate_text(output_text, language)
        
        return jsonify({'answer': output_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf_files' not in request.files:
        return jsonify({'error': 'No PDF files uploaded'}), 400
    
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files:
        return jsonify({'error': 'No PDF files selected'}), 400
    
    summaries = {}
    temp_files = []
    
    try:
        for pdf in pdf_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            pdf.save(temp_file.name)
            temp_files.append(temp_file.name)
            
            raw_text = get_pdf_text([temp_file.name])
            summary = summarize_text(raw_text)
            summaries[pdf.filename] = summary
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
            
        return jsonify({'summaries': summaries}), 200
    except Exception as e:
        # Clean up temporary files in case of error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 