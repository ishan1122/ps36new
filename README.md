# Research Paper Q&A — PS36

A Streamlit application for question answering over a research-paper PDF using RAG.

## Features
- PDF upload and page-aware text extraction
- Text splitting into overlapping chunks
- Gemini embeddings and FAISS similarity search
- Retrieval of the top 3 relevant chunks
- Gemini answers grounded in retrieved paper text
- Preset questions: Method, Dataset, Results, Limitations
- Retrieved excerpts and source PDF page references

## Requirements
Python 3.10+ and a Gemini API key from https://aistudio.google.com/

## Setup
Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
Install:
```bash
pip install -r requirements.txt
```
Copy `.env.example` to `.env`, then set your key:
```text
GOOGLE_API_KEY=your_actual_key
```
Never upload `.env` to GitHub or include it in the submitted ZIP.

Run:
```bash
streamlit run app.py
```
Upload a text-based PDF, enter the API key in the sidebar (or set it in `.env`), and click **Process research paper**.

## RAG workflow
1. PyPDFLoader extracts each page and preserves page metadata.
2. RecursiveCharacterTextSplitter creates 1,000-character chunks with 150-character overlap.
3. `models/gemini-embedding-001` creates embeddings.
4. FAISS stores vectors and retrieves the top 3 matching chunks.
5. `gemini-2.5-flash` generates an answer using retrieved context.
6. The UI displays the answer, excerpts and PDF page numbers.

## Suggested tests
Use a paper with clear method, dataset, results and limitations. Test:
1. What is the research objective?
2. Explain the methodology.
3. Which dataset was used?
4. What were the main results?
5. What limitations or future work are mentioned?

Check answers and page numbers against the original PDF.

## Limitations
- Scanned PDFs without selectable text are not OCR-processed.
- Retrieval may miss relevant information outside the top matching chunks.
- Gemini quotas and API availability may affect use.
- Verify important claims against the original paper.

Project number: PS36
Title: Research Paper Q&A
