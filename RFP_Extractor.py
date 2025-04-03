import re
import spacy
import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text
from docx import Document  # Import for handling Word documents

# Ensure the SpaCy model is downloaded
import spacy.cli
try:
    spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    return extract_text(pdf_file)

def extract_text_from_word(word_file):
    """Extract text from an uploaded Word (.docx) file."""
    doc = Document(word_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def clean_text(text):
    """Remove invalid characters from the text."""
    # Remove NULL bytes and control characters
    return re.sub(r"[\x00-\x1F\x7F]", "", text)

def extract_sentences_with_keywords(text, keywords, assigned_sentences):
    """Extract sentences that contain any of the specified keywords."""
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Use a case-insensitive search to find sentences with keywords
    keyword_pattern = re.compile(r'(?i)\b(?:' + '|'.join(keywords) + r')\b')
    matches = [
        sentence.strip()
        for sentence in sentences
        if keyword_pattern.search(sentence) and sentence.strip() not in assigned_sentences
    ]
    # Add matched sentences to the assigned list
    assigned_sentences.update(matches)
    return matches if matches else ["Not Found"]

def extract_named_entities(text, nlp, label, assigned_sentences):
    """Extract specific named entities based on label."""
    doc = nlp(text)
    matches = [
        ent.text
        for ent in doc.ents
        if ent.label_ == label and ent.text not in assigned_sentences
    ]
    # Add matched entities to the assigned list
    assigned_sentences.update(matches)
    return matches if matches else ["Not Found"]

def process_rfp(file, file_type):
    """Process the RFP document and extract key information."""
    # Extract text based on file type
    if file_type == "pdf":
        text = extract_text_from_pdf(file)
    elif file_type == "docx":
        text = extract_text_from_word(file)
    else:
        raise ValueError("Unsupported file type")

    # Clean the extracted text
    text = clean_text(text)

    # Load the SpaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Define broader keyword sets for each section
    scope_keywords = [
        "Scope", "Description", "Objective", "Goals",
        "Deliverables", "Statement of Work"
    ]

    methodology_keywords = [
        "Methodology", "Approach", "Strategy", "Plan", "Implementation", "Execution",
        "Framework", "Process", "Techniques", "Procedures",
        "Workflow", "Guidelines", "Steps"
    ]

    eligibility_keywords = [
        "Eligibility", "Eligible", "Applicants", "Who can apply", "Requirements",
        "Qualifications", "Criteria", "Conditions", "Restrictions", "Target Audience",
        "Vendor Requirements", "Compliance", "Certification"
    ]

    budget_keywords = [
        "Budget", "Funding", "Cost", "Financial", "Expenses", "Price", "Pricing",
        "Allocation", "Payment Terms", "Compensation", "Fee", "Proposal Cost",
        "Estimated Budget", "Financial Plan", "Cost Breakdown"
    ]

    deadline_keywords = [
        "Deadline", "Submission", "Due Date", "Timeline", "Schedule",
        "Important Dates", "Deliverable Dates", "Completion Date", "Timeframe",
        "Response Date", "Proposal Deadline", "Final Date", "Submission Window"
    ]

    selection_process_keywords = [
        "Selection", "Evaluation", "Criteria", "Process", "Weighting",
        "Judging", "Metrics", "Assessment", "Decision", "Review",
        "Proposal Review", "Selection Process", "Evaluation Process", "Scoring Rubric"
    ]
    
    # Track assigned sentences to avoid duplication
    assigned_sentences = set()
    
    extracted_info = {
        "Category": [
            "Scope of Work", "Methodology", "Eligibility", "Budget", "Deadlines",
            "Selection Process"
        ],
        "Details": [
            "\n".join(extract_sentences_with_keywords(text, scope_keywords, assigned_sentences)),
            "\n".join(extract_sentences_with_keywords(text, methodology_keywords, assigned_sentences)),
            "\n".join(extract_sentences_with_keywords(text, eligibility_keywords, assigned_sentences)),
            "\n".join(extract_sentences_with_keywords(text, budget_keywords, assigned_sentences)),
            "\n".join(extract_named_entities(text, nlp, "DATE", assigned_sentences)),
            "\n".join(extract_sentences_with_keywords(text, selection_process_keywords, assigned_sentences)),
        ]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(extracted_info)
    return df

def save_to_word(df, output_path):
    """Save extracted information to a Word document."""
    doc = Document()
    doc.add_heading("RFP Extracted Information", level=1)

    for index, row in df.iterrows():
        doc.add_heading(row["Category"], level=2)
        doc.add_paragraph(row["Details"], style="BodyText")

    # Save the Word document
    doc.save(output_path)

# Streamlit App
st.title("RFP Key Information Extractor")
st.write("Upload an RFP document (PDF or Word) to extract key details such as Scope, Eligibility, Budget, Deadlines, Selection Process, and Methodology.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type not in ["pdf", "docx"]:
        st.error("Unsupported file type. Please upload a PDF or Word document.")
    else:
        df = process_rfp(uploaded_file, file_type)
        
        # Display extracted information
        st.write("### Extracted Information")
        st.dataframe(df)
        
        # Save to Word and provide download link
        output_word = "rfp_extracted_info.docx"
        save_to_word(df, output_word)
        
        with open(output_word, "rb") as f:
            st.download_button("Download Extracted Info as Word", f, file_name="rfp_extracted_info.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")