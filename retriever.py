import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import CrossEncoderReranker
from dotenv import load_dotenv

load_dotenv()

# Initialize core models once when the file loads
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Load the Cross-Encoder for the Two-Stage Re-ranking
cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=4)

def get_answer_and_citations(user_query, target_chapter="All Chapters"):
    """Queries the database and returns a tuple: (Answer_String, List_of_Citations)"""
    
    # 1. Connect to the database
    if not os.path.exists("./chroma_db"):
        return "Database not found. Please upload and process PDFs first.", []
        
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 2. Apply the strict Metadata Pre-Filter
    search_kwargs = {"k": 15} # Fetch top 15 initially
    if target_chapter and target_chapter != "All Chapters":
        search_kwargs["filter"] = {"chapter": target_chapter}
        
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # 3. Wrap in the Re-ranker
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # 4. Prompt Engineering
    system_prompt = (
        "You are an expert analytical assistant. Use the provided context excerpts to answer the question. "
        "If the answer is not contained in the context, say 'I cannot find this in the uploaded documents.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Execute Chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(advanced_retriever, qa_chain)
    
    response = rag_chain.invoke({"input": user_query})
    
    # 6. Extract exact citations to return to the UI
    citations = []
    for doc in response["context"]:
        citations.append({
            "file": doc.metadata.get("source", "Unknown"),
            "chapter": doc.metadata.get("chapter", "Unknown"),
            "text": doc.page_content[:200] + "..." # Snippet to prove the AI read it
        })
        
    return response["answer"], citations