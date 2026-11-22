# PS36 Procedure Report — preparation notes

Fill in your actual name, roll number, class, GitHub URL and date. Add screenshots from your own run. Do not claim tests you have not performed.

## Problem statement
Students often need to read many research papers for literature reviews. This project allows a user to upload one research-paper PDF and ask questions about its methodology, dataset, results and limitations. The application retrieves relevant passages and uses an LLM to formulate an answer with source-page references.

## Tools and model
Python, Streamlit, PyPDFLoader, RecursiveCharacterTextSplitter, Google Gemini embeddings (`models/gemini-embedding-001`), FAISS, and Gemini chat model (`gemini-2.5-flash`).

## Procedure
1. Created a Python project and installed dependencies.
2. Obtained a Gemini API key and configured it through an environment variable or the app sidebar.
3. Uploaded a text-based research-paper PDF.
4. Extracted page-level text and split it into overlapping chunks.
5. Generated embeddings and stored them in FAISS.
6. Retrieved the three most relevant chunks for each question.
7. Sent retrieved context and the question to Gemini with instructions to answer only from the paper.
8. Displayed the answer, retrieved excerpts and PDF page numbers in Streamlit.
9. Tested questions about the objective, method, dataset, results and limitations.

## Prompt used
You are an academic research paper assistant. Answer only using retrieved excerpts from the uploaded research paper. Do not use outside knowledge or invent facts, methods, datasets, metrics or citations. Explain clearly while preserving academic accuracy. Cite supplied PDF page numbers. If excerpts do not contain enough information, say that the information was not found in the uploaded paper.

## Prompt improvement
The prompt explicitly prohibits outside knowledge and invented citations, and requires a not-found response when retrieved context is insufficient.

## Results, challenges and conclusion
Insert screenshots and describe the actual paper and test questions used. Record errors you encountered and how you resolved them. State what worked in your tests, current limitations (for example, scanned PDFs need OCR), and one future improvement.
