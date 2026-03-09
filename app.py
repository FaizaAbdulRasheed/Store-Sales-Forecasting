import streamlit as st

st.set_page_config(page_title="AI Project Portfolio", layout="wide")

st.title("AI / ML Project Portfolio")
st.write("Live demonstrations of AI, LLM, and Machine Learning projects.")

st.divider()

# Project 1
st.header("1. Intelligent Technical Support Agent")
st.write("""
RAG | LangChain | Pinecone | Llama-3

• Built a Retrieval Augmented Generation system to answer technical questions from product documentation.  
• Implemented hybrid search combining semantic embeddings and keyword retrieval.  
• Integrated Llama-3 for context-aware responses.
""")

st.divider()

# Project 2
st.header("2. Financial 10-K Analyst")
st.write("""
RAG | Document Intelligence | LLM Summarization

• Developed a document intelligence system to analyze SEC 10-K filings.  
• Implemented PDF parsing, document chunking, and embedding retrieval.  
• Generates cited answers and summaries from long financial reports.
""")

st.divider()

# Project 3
st.header("3. AI Legacy Code Modernization Assistant")
st.write("""
LLM Applications | Code Intelligence | RAG

• Built an AI tool to analyze and explain legacy codebases.  
• Uses vector retrieval for contextual code understanding.  
• Generates documentation and modernization suggestions.
""")

st.divider()

# Project 4
st.header("4. Credit Risk Scoring Engine")
st.write("""
Machine Learning | XGBoost | SHAP

• Built a loan default prediction model using the LendingClub dataset.  
• Achieved ROC-AUC 0.88 and F1 score 0.83.  
• Implemented SHAP explainability to identify key risk factors.
""")

st.divider()

st.success("Projects are currently being deployed. Full interactive demos will be updated soon.")
