# Supreet's PDFChatbot

A Streamlit web application that allows users to chat with PDF documents using Google's Gemini AI. The app supports multiple languages and can provide summaries of PDF documents.

## Features

- Upload and process multiple PDF files
- Ask questions about the content of your PDFs
- Get responses in multiple languages
- Generate summaries of PDF documents
- Download summaries as text files

## Requirements

- Python 3.8 or higher
- Google API Key for Gemini AI

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API Key:
   ```
   GOOGLE_API_KEY="your-api-key-here"
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`
3. Upload your PDF files
4. Ask questions or generate summaries

## Deployment

This app can be deployed on Streamlit Cloud. Follow these steps:

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" and select your repository
5. Set the main file path to `app.py`
6. Add your environment variables (GOOGLE_API_KEY)
7. Deploy!

## License

This project is licensed under the MIT License. 