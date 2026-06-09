import gradio as gr
import os
import shutil
from retriever import get_answer_and_citations
from data_processor import ingest_pdfs_bulletproof # Import your ingestion logic!

# --- HELPER FUNCTIONS ---
def process_uploaded_files(uploaded_files):
    """Moves Gradio temp files to our documents folder and triggers the AI parser."""
    if not uploaded_files:
        return "No files uploaded."
        
    # Ensure the target directory exists
    os.makedirs("./documents", exist_ok=True)
    
    # Copy files from Gradio's temp folder to our local directory
    for file in uploaded_files:
        filename = os.path.basename(file.name)
        shutil.copy(file.name, f"./documents/{filename}")
        
    # Trigger your Unstructured data processor!
    try:
        ingest_pdfs_bulletproof(pdf_directory="./documents")
        return f"Successfully processed {len(uploaded_files)} documents into the database!"
    except Exception as e:
        return f"Error processing documents: {str(e)}"

def chat_interface(message, history, chapter_filter):
    """Handles the user query and formats the chat history + citations."""
    answer, citations = get_answer_and_citations(message, chapter_filter)
    
    # Format the citations neatly for the Gradio chat window
    if citations:
        citation_text = "\n\n**Sources Cited:**\n"
        for i, c in enumerate(citations):
            citation_text += f"- **[{i+1}] {c['file']}** (Chapter: {c['chapter']})\n"
            citation_text += f"  > *\"{c['text']}\"*\n"
        
        final_response = answer + citation_text
    else:
        final_response = answer
        
    return final_response

# --- GRADIO UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enterprise Document Intelligence")
    gr.Markdown("Upload multiple PDFs, parse their structure, and chat with exact citations.")
    
    with gr.Row():
        # LEFT COLUMN: Data Ingestion
        with gr.Column(scale=1):
            gr.Markdown("### 1. Knowledge Base")
            file_upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
            process_btn = gr.Button("Build Vector Database", variant="primary")
            status_text = gr.Textbox(label="System Status", interactive=False)
            
            # Link the button to the ingestion function
            process_btn.click(
                fn=process_uploaded_files,
                inputs=[file_upload],
                outputs=[status_text]
            )
            
            gr.Markdown("### 2. Search Filters")
            # In a full app, you would read the Chroma DB to dynamically populate this list
            chapter_dropdown = gr.Dropdown(
                choices=["All Chapters", "Introduction", "Zusammenfassung", "Erklärung", "Conclusion"],
                value="All Chapters",
                label="Restrict Search To Chapter:"
            )

        # RIGHT COLUMN: Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("### 3. Secure Chat")
            # Gradio's ChatInterface handles the history state automatically
            chat = gr.ChatInterface(
                fn=chat_interface,
                additional_inputs=[chapter_dropdown],
                undo_btn=None,
                clear_btn="Clear Chat"
            )

if __name__ == "__main__":
    # Launch the server
    demo.launch(server_name="0.0.0.0", server_port=7860)