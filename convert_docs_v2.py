import os
import glob
from pathlib import Path
import fitz  # PyMuPDF
from markdownify import markdownify as md

def convert_pdf_to_markdown(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = glob.glob(str(input_path / "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting conversion using PyMuPDF...")

    for pdf_file_path in pdf_files:
        pdf_file = Path(pdf_file_path)
        print(f"Converting: {pdf_file.name}")

        try:
            doc = fitz.open(pdf_file_path)
            full_text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Get text as HTML to preserve some structure for markdownify
                html_content = page.get_text("html")
                full_text += html_content

            doc.close()

            # Convert HTML to Markdown
            markdown_text = md(full_text)

            # Define output filename
            output_file = output_path / f"{pdf_file.stem}.md"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            print(f"Successfully converted to {output_file.name}")

        except Exception as e:
            print(f"Error converting {pdf_file.name}: {e}")

if __name__ == "__main__":
    # Configuration
    DOCS_DIR = "vllm-neuron-docs"
    OUTPUT_DIR = "vllm-neuron-docs/markdown_output"

    convert_pdf_to_markdown(DOCS_DIR, OUTPUT_DIR)
    print("\nConversion process finished.")
