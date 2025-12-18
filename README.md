# 🎓 Constructor University Program Selection AI  
**AI Assistant for Academic Program Discovery**

---

## Overview

The **Constructor University Program Selection AI** is an evidence-grounded, retrieval-augmented AI system designed to help prospective students explore, compare, and understand **Bachelor’s and Master’s programs at Constructor University**.

Unlike generic conversational AI systems, this assistant **does not guess or hallucinate**. Every response is strictly grounded in **official university sources**, including:

- Program Handbook PDFs (Bachelor & Master)
- Structured Program Facts (Excel datasets)
- Official Constructor University website pages

Students can ask natural-language questions and receive **transparent, citation-backed answers**, ensuring trust, correctness, and academic reliability.

---

## What This Project Solves

### Key Challenges
- Information spread across PDFs, Excel sheets, and multiple web pages  
- Difficulty comparing programs across fees, duration, language, and requirements  
- Risk of outdated or unofficial information  
- Hallucinations from general-purpose AI tools  

### Our Solution
- Centralized, curated knowledge base using **only official documents**
- Strict **evidence-only** answer generation
- Clear citations for every factual claim
- Source-controlled querying (PDF-only, Excel-only, Web-only)
- Reliable program-to-program comparisons

---

## Core Features

### Evidence-Based Question Answering
- Answers are generated strictly from indexed documents  
- If evidence is missing, the system responds:
  > *“I don’t know based on the provided documents.”*

### Multi-Source Knowledge Base
- **PDFs**: Official Bachelor & Master program handbooks  
- **Excel**: Structured program facts (fees, duration, tests, contacts)  
- **Web**: Admissions rules, English requirements, application steps  

### Retrieval-Augmented Generation (RAG)
- Semantic retrieval using **Qdrant vector database**
- Context assembly from top-ranked evidence chunks
- Controlled LLM response generation with strict system prompts

### Program-Aware Queries
- Automatic resolution of program names and abbreviations (e.g., CS, DSSB, AST)
- Program-specific filtering to prevent cross-program mixing
- Accurate side-by-side comparisons

### Transparent Citations
- Each answer includes:
  - Source type (PDF / Excel / Web)
  - Program name or ID
  - Short quoted evidence snippet
- Evidence is deduplicated and capped for clarity

### 🖥️ Interactive Streamlit Interface
- Chat-based Q&A
- Program browsing tables
- Dedicated tabs for:
  - Tuition & Fees
  - English Requirements
  - Application Documents
  - Official Contacts

---

## Technical Architecture

User Question
   ↓
Embedding Model (OpenAI)
   ↓
Qdrant Vector Search
   ↓
Source Filtering (PDF / Excel / Web)
   ↓
Context Assembly (Top-K chunks)
   ↓
LLM Answer Generation (Strict RAG rules)
   ↓
Validation & Evidence Limiting
   ↓
Final Answer + Citations

---


### Core Components

#### Vector Database - Qdrant
- Stores embeddings for PDFs, Excel program cards, and web pages
- Deterministic point IDs ensure stable re-indexing
- Scales efficiently as the corpus grows

#### LLM Models
- **Embedding model**: `text-embedding-3-large` (3072 dimensions)
- **Chat model**: `gpt-4.1-mini`
- Separation of retrieval and generation improves reliability and performance

#### Intelligent Chunking
- Documents split into semantically meaningful chunks
- Overlapping windows preserve context
- Excel rows converted into structured “program cards” with sections:
  - Overview
  - Admissions
  - Fees
  - Scholarships & Contacts

---

## Curated Reference Library

The knowledge base includes **only official Constructor University materials**:

### Sources Used
- Program Handbooks (PDF): All Bachelor and Master programs  
- Program Facts (Excel): Fees, duration, credits, language, tests, scholarships, contacts  
- University Website: Admissions process, English requirements, documents  

### Sources Excluded
- Blogs, forums, rankings, student opinions, or third-party summaries

---

## ⚙️ How It Works (End-to-End)

1. **Indexing Phase**
   - PDFs are parsed, cleaned, chunked, embedded, and stored in Qdrant  
   - Excel files are transformed into structured program chunks  
   - Website pages are crawled, cleaned, cached, chunked, and embedded  

2. **Query Phase**
   - User question is embedded  
   - Qdrant retrieves top semantic matches  
   - Optional source filters are applied  
   - Program resolution narrows scope when relevant  

3. **Answer Generation**
   - Retrieved evidence is passed to the LLM  
   - System prompt enforces no guessing, precise terminology, and balanced comparisons  
   - Answer is generated strictly from provided context  

4. **Validation Layer**
   - Numeric claims (fees, scores) are checked against citations  
   - Evidence presence is enforced  
   - Confidence level is computed  

5. **User Interface Rendering**
   - Clean, readable answer  
   - Evidence box with citations  
   - Warnings shown when confidence is low  

---

## Intended Use

This system is designed for:
- Prospective Bachelor’s and Master’s applicants  
- International students comparing programs  
- Admissions guidance demonstrations  
- Academic AI system evaluation and coursework  

It is **not** intended to replace official admissions offices or provide legal/visa advice.

---

## Limitations

- Only answers questions covered by indexed documents  
- Cannot infer or estimate missing information  
- Website ingestion depends on crawl configuration  
- Program updates require re-indexing  

---

## Future Directions

- Multi-university support (EU / DAAD programs)
- Automatic re-indexing on handbook updates
- Visual program comparison dashboards
- Personalized application checklists
- Role-based access for admissions staff

---




