# BankBot — AI Chatbot for Banking FAQs

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![AI/NLP](https://img.shields.io/badge/AI–NLP-green)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![LLM](https://img.shields.io/badge/LLM-Configurable-red)](https://en.wikipedia.org/wiki/Language_model)
[![Status](https://img.shields.io/badge/Status-Prototype-yellow)]
[![License](https://img.shields.io/badge/License-MIT-blue)]

BankBot is an **AI-powered chatbot** designed to help users interactively answer **frequently asked questions related to banking**. This project was developed as part of an **Infosys certification / internship exercise**, leveraging **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** to generate intelligent responses to user queries. ([LinkedIn][1])

---

## 📌 Project Description

BankBot is a conversational AI system that understands natural language queries about common banking topics (e.g., services, products, processes) and provides accurate answers. It combines NLP preprocessing, configurable LLM-based text generation, and prompt engineering to deliver context-aware responses.

Unlike rule-based chatbots, BankBot uses transformer-based language models that can be configured with different LLM backends (open-source or API-driven) to suit accuracy and deployment needs.

---

## 🚀 Features

* 🧠 Natural Language Understanding of banking FAQs
* 💬 AI-driven conversational responses
* ⚙️ **Configurable LLM** backend for flexible model choices
* 🔄 Prompt engineering for enhanced response quality
* 📊 Easy extension to more topics or dialogue flows
* 🧪 Modular and extensible codebase

---

## 💡 Techniques Used

### Natural Language Processing (NLP)

BankBot preprocesses user queries using standard NLP techniques, such as:

* Tokenization
* Normalization
* Intent recognition
* Entity extraction

These steps help the model better understand banking queries before generation.

### Prompt Engineering

BankBot uses prompt templates to:

* Structure user queries for the LLM
* Provide context (e.g., FAQ context)
* Improve accuracy of generated answers

This ensures responses are relevant and domain-specific.

### LLM-based Text Generation

BankBot integrates transformer-based **Large Language Models** (such as GPT-series, BERT-based models for embeddings, or other configurable LLMs). The model generates or retrieves responses based on training and prompt context.

🔁 **Configurable LLM:** You can switch between different transformer-based models depending on requirements (e.g., open-source model vs API-hosted model).

---

## 🧰 Tech Stack

### Programming Language

* Python 3.8+

### Libraries / Frameworks

* `transformers` — for LLMs and model integration
* `NLTK` / `spaCy` — for NLP preprocessing
* Flask / FastAPI — optional backend server for chat interface
* SQLite — local knowledge or FAQ storage
* Jupyter Notebooks — experiments and training materials

### AI / ML technologies

* NLP preprocessing
* Embeddings and semantic understanding
* LLM integration for text generation

### LLM Details

BankBot uses **transformer-based language models** for generating responses.
These can be configured to use:

* Local open-source models (e.g., GPT-2/3 style models via `transformers`)
* API-driven models (OpenAI GPT, Claude, etc.)
* Retrieval–augmented generation setups

Model configuration parameters (model name, API keys, prompt templates) are stored in a config file for easy adjustment.

---

## 📁 Project Structure

```
.
├── nlu_engine/                # NLP and intent processing modules  
├── database/                  # FAQ database and storage  
├── main_app1.py               # Application entry point  
├── Milestone_2.py             # Experimental LLM integration code  
├── bankbot.db                 # Pre-built SQLite knowledge store  
├── Training Materials/        # Notebooks, datasets, guides  
├── Experiments/               # Prototypes and model experiments  
└── README.md                 # Project documentation
```

---

## 🛠 Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Aitha-Rohith-Kumar/Infosys-_Project-BankBot-AI-Chatbot-for-Banking-FAQs
   cd Infosys-_Project-BankBot-AI-Chatbot-for-Banking-FAQs
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   # .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure LLM**

   * Create a config file (e.g., `config.json`) with your chosen model and API keys.
   * Example:

     ```json
     {
       "llm_model": "gpt-3.5-turbo",
       "api_key": "YOUR_API_KEY"
     }
     ```

---

## ▶️ Running Locally

1. **Initialize database (if applicable)**

   ```bash
   python main_app1.py --init_db
   ```

2. **Start the chatbot service**

   ```bash
   python main_app1.py
   ```

3. **Interact with BankBot**

   * Via terminal prompt
   * Or through a frontend interface (if implemented)

> You may need to adjust environment variables for API keys or model configs before running.

---

## 🧾 Certification Use Case

This project was completed as part of an **Infosys certification / internship program**, focusing on real-world AI/NLP use cases. BankBot demonstrates your ability to:

* Build an NLP-driven AI assistant
* Integrate transformer-based LLMs
* Apply prompt engineering techniques
* Deploy a configurable and modular system

Use this project in your portfolio or certification submission to showcase your skills in **AI, NLP, and LLM integration**. ([LinkedIn][1])

---


