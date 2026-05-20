import os
from langchain_unstructured import UnstructuredLoader
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def processing_pdfs(pdf_directory="./documents"):
    print("Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    all_langchain_chunks = []
    
    print(f"Scanning {pdf_directory} for PDFs...")
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            
            print(f"Parsing {filename} (This may take a moment)...")
            loader = UnstructuredLoader(
                filepath, 
                strategy="hi_res", # uses layout detection to find headers and tables
                mode="elements"
            )
            raw_elements = loader.load()
            
            structured_chunks = chunk_by_title(
                [el.metadata["unstructured_element"] for el in raw_elements], 
                max_characters=1000, 
                combine_text_under_n_chars=200
            )
            
            chapters_found = []
            for chunk in structured_chunks:
                # Unstructured automatically captures the parent header it found.
                # If it's under "Chapter 3: Deployment", that string is saved here.
                section_title = chunk.metadata.to_dict().get("parent_id", "Unknown Section")
                chapters_found.append(section_title)
                
                # Convert the Unstructured chunk back into a LangChain Document
                doc = Document(
                    page_content=chunk.text,
                    metadata={
                        "source": filename,
                        # We dynamically inject the real detected header/chapter
                        "chapter": section_title 
                    }
                )
                all_langchain_chunks.append(doc)
            
            unique_chapters = sorted(set(chapters_found))
            print(f"Built {len(structured_chunks)} chunks for {filename}.")
            if unique_chapters:
                print(f"Detected chapters in {filename}: {unique_chapters}")
            else:
                print(f"No chapter titles detected in {filename}.")

    print("Building Chroma Vector Database...")
    vector_store = Chroma.from_documents(
        documents=all_langchain_chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("Success. Database built with real structural metadata.")

if __name__ == "__main__":
    if not os.path.exists("./documents"):
        os.makedirs("./documents")
        print("Created './documents' folder. Please add PDFs and run again.")
    else:
        ingest_pdfs_intelligently()
