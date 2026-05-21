from unstructured.partition.pdf import partition_pdf

# Put the exact path to your problematic PDF here
pdf_path = "./documents/Final_Thesis_KIT.pdf"

print(f"Analyzing {pdf_path}...\n")

# We run the raw partitioner without chunking to see the base elements
loader = UnstructuredLoader(
                filepath=pdf_path, 
                strategy="hi_res", 
                chunking_strategy="by_title",
                pdf_infer_table_structure=True, # Forces deeper visual analysis
                languages=["eng", "deu"],       # English and German (for "Erklärung")
                max_characters=1000, 
                combine_text_under_n_chars=200
            )

# Print the first 20 elements to see how the AI classified them
for i, element in enumerate(elements[:20]):
    # element.category will be things like "Title", "NarrativeText", "ListItem"
    print(f"Type: {element.category:<15} | Text: {element.text[:60]}...")