import os
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def process_pdfs(pdf_directory="./documents"):
    print("Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    all_langchain_chunks = []
    
    print(f"Scanning {pdf_directory} for PDFs...")
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            
            print(f"Parsing and chunking {filename}...")
            loader = UnstructuredLoader(
                filepath, 
                strategy="hi_res", 
                chunking_strategy="by_title",
                pdf_infer_table_structure=True, # Forces deeper visual analysis
                languages=["eng", "deu"],       # English and German (for "Erklärung")
                max_characters=1000, 
                combine_text_under_n_chars=200
            )
            
            # It loads AND chunks intelligently in one single step
            structured_chunks = loader.load()
            
            for chunk in structured_chunks:
                # Add our source filename to the metadata
                chunk.metadata["source"] = filename
                
                # Unstructured automatically saves the section title in 'parent_id' or 'category'
                # We rename it to "chapter" so our Chroma filter works perfectly
                section_title = chunk.metadata.get("section", "Unknown Section")
                chunk.metadata["chapter"] = section_title

                print(f"section_title: {section_title}")
                
                all_langchain_chunks.append(chunk)
            
            print(f"Built {len(structured_chunks)} smart chunks for {filename}.")

    print("Building Chroma Vector Database...")
    vector_store = Chroma.from_documents(
        documents=all_langchain_chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("Success! Database built with real structural metadata.")

if __name__ == "__main__":
    if not os.path.exists("./documents"):
        os.makedirs("./documents")
        print("Created './documents' folder. Please add PDFs and run again.")
    else:
        process_pdfs()