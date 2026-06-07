import os
# Fix the import to include UnstructuredLoader from LangChain
from langchain_unstructured import UnstructuredLoader

import torch

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# Put the exact path to your problematic PDF here
pdf_path = "./documents/Final_Thesis_KIT.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: Could not find file at {pdf_path}. Please check the filename.")
else:
    print(f"Analyzing {pdf_path} using high-resolution layout analysis...\n")

    # Initialize the loader exactly how it runs in your main ingestion pipeline
    loader = UnstructuredLoader(
        pdf_path, 
        strategy="hi_res", 
        chunking_strategy="by_title",
        max_characters=1000, 
        combine_text_under_n_chars=200
    )

    # Load the chunked documents
    chunks = loader.load()

    # Print the first 10 chunks to see what metadata keys are being produced
    for i, chunk in enumerate(chunks[:10]):
        # Get the section or category tracking keys
        section = chunk.metadata.get("section", "None")
        category = chunk.metadata.get("category", "None")
        
        print(f"Chunk [{i+1}]")
        print(f"  -> Detected Section: '{section}'")
        print(f"  -> Element Category: '{category}'")
        print(f"  -> Content Snippet: \"{chunk.page_content[:80]}...\"\n")
        
    # Print available metadata keys to inspect what the dictionary contains
    if chunks:
        print("Available Metadata Keys in your LangChain version:")
        print(list(chunks[0].metadata.keys()))
