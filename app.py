from flask import Flask, render_template, request, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pdfplumber
import docx2txt
import requests
from nltk.tokenize import sent_tokenize
import html

app = Flask(__name__)
app.secret_key = 'd082e9e0d30308340df87d13031e3b89'  # Add a secret key for session management

# Function to fetch text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        return None

# Function to get original texts from files
def get_original_texts():
    # Define your list of original texts here
    pdf_files = ["C:\\Users\\krish\\Downloads\\final\\Task Details.pdf", "C:\\Users\\krish\\Downloads\\final\\UNIT-3 FINAL.pdf", "C:\\Users\\krish\\Downloads\\final\\Siddartha's Resume.pdf"]
    docx_files = ["C:\\Users\\krish\\Downloads\\final\\CV UNIT-1.docx", "C:\\Users\\krish\\Downloads\\final\\CV UNIT-2.docx", "C:\\Users\\krish\\Downloads\\final\\MST_Internship_Doc.docx"]
    txt_files = ["C:\\Users\\krish\\Downloads\\final\\Data Dictionary.txt", "C:\\Users\\krish\\Downloads\\final\\Deloitte coverletter.txt"]

    original_texts = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            original_texts.append(text)

    for docx_file in docx_files:
        text = extract_text_from_docx(docx_file)
        if text:
            original_texts.append(text)

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            if text:
                original_texts.append(text)

    return original_texts

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return ''

# Function to extract text from DOCX files
def extract_text_from_docx(file):
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        print("Error extracting text from DOCX file:", e)
        return ''


# Function to calculate similarity scores
def calculate_similarity(input_text, original_texts):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([input_text] + original_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

# Function to determine text type
def determine_text_type(similarity_scores, threshold):
    if any(score >= threshold for score in similarity_scores):
        return "original"
    else:
        return "copied"

# Function to find copied sentences
def find_copied_sentences(input_text, original_texts, threshold):
    copied_sentences = []
    input_sentences = sent_tokenize(input_text)
    print("Input Sentences:", input_sentences)
    for sentence in input_sentences:
        for original_text in original_texts:
            if sentence in original_text:
                copied_sentences.append(sentence)
                break
    return copied_sentences

# Function to highlight copied sentences
def highlight_copied_sentences(input_text, copied_sentences):
    highlighted_text = input_text
    for sentence in copied_sentences:
        highlighted_text = re.sub(re.escape(sentence), f'<span class="copied">{html.escape(sentence)}</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form.get('input_type')

        if input_type == 'text':
            input_text = request.form.get('input_text', '')
        elif input_type == 'file':
            file = request.files.get('file')
            if file:
                if file.filename.endswith('.txt'):
                    input_text = file.read().decode('utf-8')
                elif file.filename.endswith('.pdf'):
                    input_text = extract_text_from_pdf(file)
                elif file.filename.endswith('.docx'):
                    input_text = extract_text_from_docx(file)
                else:
                    flash('Unsupported file format. Please upload a TXT, PDF, or DOCX file.')
                    return render_template('index.html')
            else:
                flash('No file uploaded.')
                return render_template('index.html')
        else:
            flash('Invalid input type.')
            return render_template('index.html')

        threshold = 0.7  # Adjust the threshold as needed
        original_texts = get_original_texts()
        print("Original Texts:", original_texts)

        # Calculate similarity scores
        similarity_scores = calculate_similarity(input_text, original_texts)
        print("Similarity Scores:", similarity_scores)

        # Determine text type based on similarity scores
        text_type = determine_text_type(similarity_scores, threshold)

        # Find copied sentences
        copied_sentences = find_copied_sentences(input_text, original_texts, threshold)
        print("Copied Sentences:", copied_sentences)

        # Highlight copied sentences in input text
        highlighted_text = highlight_copied_sentences(input_text, copied_sentences)

        # Calculate metrics
        num_copied_sentences = len(copied_sentences)
        num_input_sentences = len(sent_tokenize(input_text))
        percentage_plagiarized = (num_copied_sentences / num_input_sentences) * 100 if num_input_sentences > 0 else 0

        return render_template('results.html', text_type=text_type, num_copied_sentences=num_copied_sentences,
                               percentage_plagiarized=percentage_plagiarized, copied_sentences=copied_sentences,
                               highlighted_text=highlighted_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
