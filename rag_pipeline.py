from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from vector_database import retrieve_docs as retrieve_filtered_docs
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
load_dotenv()

print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Debug log
    print("ENV NOT FOUND — current working directory:", os.getcwd())
    print("List of files here:", os.listdir(os.getcwd()))
    raise ValueError("GROQ_API_KEY is missing! Set it in your .env file or environment variables.")

# Get the API key
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# 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

# Check if API key is set
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Set it in your .env file or environment variables.")

# Step1: Setup LLM (Use DeepSeek R1 with Groq)
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Step2: Retrieve Docs
def retrieve_docs(query, file_name):
    return retrieve_filtered_docs(query, file_name)

def get_context(documents):
    # Include page reference in each chunk
    context = "\n\n".join([
        f"{doc.page_content}\n(Source: Page {doc.metadata.get('page', 'Unknown')})"
        for doc in documents
    ])
    return context


# Step3: Answer Question with Follow-Up Support
custom_prompt_template = """
Use the pieces of information provided in the context and previous conversation history to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Previous Conversation:
{history}

Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query, history=""):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context, "history": history})
    return response

# Step4: Summarization Function
def summarize_document(documents):
    context = get_context(documents)
    
    summary_prompt = """
You are a legal AI assistant. Summarize the document below in a formal structure. After each key point or section, include a citation like (Source: Page X) to indicate where the information was retrieved from. Avoid adding internal thoughts or explanations.

Document:
{context}

Return the summary in the following format:

### Document Summary

1. **Section Title**: Description... (Source: Page X)
2. **Section Title**: Description... (Source: Page Y)
"""
    prompt = ChatPromptTemplate.from_template(summary_prompt)
    chain = prompt | llm_model
    return chain.invoke({"context": context})

def clean_summary_output(response):
    if hasattr(response, 'content'):
        return response.content.encode().decode('unicode_escape').replace("\\'", "'").strip()
    return str(response).strip()


# Step5: Generate Downloadable Report using ReportLab
def generate_report(user_queries, ai_responses):
    pdf_path = "AI_Lawyer_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Lawyer Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Below is a record of your conversation with AI Lawyer.")
    
    y = 700
    max_width = 450  # Maximum width for text before wrapping
    line_height = 15
    
    for question, answer in zip(user_queries, ai_responses):
        c.setFont("Helvetica-Bold", 12)
        q_lines = simpleSplit(f"Q: {question}", "Helvetica-Bold", 12, max_width)
        a_lines = simpleSplit(f"A: {answer}", "Helvetica", 12, max_width)
        
        for line in q_lines:
            c.drawString(100, y, line)
            y -= line_height
        
        c.setFont("Helvetica", 12)
        for line in a_lines:
            c.drawString(100, y, line)
            y -= line_height
        
        y -= 20  # Extra space between Q&A
        
        if y < 50:  # Prevent text from overflowing
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750
    
    c.save()
    return pdf_path