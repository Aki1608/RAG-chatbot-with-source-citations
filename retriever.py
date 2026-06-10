import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Initialize baseline structural models
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Load the Cross-Encoder directly using sentence-transformers (Bypasses LangChain splits!)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_answer_and_citations(user_query, target_chapter="All Chapters"):
    """Queries Chroma, runs a manual cross-encoder re-rank, and returns the answer and citations."""
    
    if not os.path.exists("./chroma_db"):
        return "Database not found. Please upload and process PDFs first.", []
        
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 1. Base Retrieval with Metadata Pre-Filtering
    search_kwargs = {"k": 15}
    if target_chapter and target_chapter != "All Chapters":
        search_kwargs["filter"] = {"chapter": target_chapter}
        
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs = base_retriever.invoke(user_query)
    
    if not retrieved_docs:
        return "I cannot find any matches in the selected documents/chapters.", []

    # 2. Manual Two-Stage Re-ranking via Cross-Encoder (Highly precise, zero LangChain imports)
    pairs = [[user_query, doc.page_content] for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    
    # Match docs to scores and sort from highest relevance to lowest
    scored_docs = list(zip(retrieved_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top 4 chunks
    top_docs = [doc for doc, score in scored_docs[:4]]
    
    # 3. Format Context String manually for the LLM
    context_str = "\n\n".join([
        f"Source Document: {doc.metadata.get('source', 'Unknown')}\n"
        f"Chapter/Section: {doc.metadata.get('chapter', 'Unknown')}\n"
        f"Excerpt: {doc.page_content}"
        for doc in top_docs
    ])
    
    # 4. Standard Core Prompt & Direct Execution
    system_prompt = (
        "You are an expert analytical assistant. Use the provided context excerpts to answer the question. "
        "If the answer is not contained in the context, say 'I cannot find this in the uploaded documents.'\n\n"
        "Context:\n{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    messages = prompt_template.format_messages(context=context_str, input=user_query)
    
    # Generate the grounded response directly via the LLM
    response = llm.invoke(messages)
    answer = response.content
    
    # 5. Compile precise citation dictionaries for Gradio
    citations = []
    for doc in top_docs:
        citations.append({
            "file": doc.metadata.get('source', 'Unknown'),
            "chapter": doc.metadata.get('chapter', 'Unknown'),
            "text": doc.page_content[:200] + "..."
        })
        
    return answer, citations