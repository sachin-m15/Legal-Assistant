import os
import json
import fitz  # type: ignore # PyMuPDF


def convert_pdfs_to_txt(source_dir: str, dest_dir: str):
    """Converts all PDF files in a source directory to text files."""
    print("--- Converting PDF files to text... ---")
    if not os.path.exists(source_dir):
        print(
            f"Source directory '{source_dir}' not found. Please place your PDF files here."
        )
        return

    for filename in os.listdir(source_dir):
        if filename.endswith(".pdf") or filename.endswith(".PDF"):
            pdf_path = os.path.join(source_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(dest_dir, txt_filename)

            try:
                doc = fitz.open(pdf_path)
                text_content = ""
                for page in doc:
                    text_content += page.get_text()

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_content)

                print(f"Successfully converted '{filename}' to '{txt_filename}'.")
            except Exception as e:
                print(f"Error converting '{filename}': {e}")


def convert_json_to_txt(source_dir: str, dest_dir: str):
    """Converts all JSON files with Q&A pairs to a single text file."""
    print("--- Converting JSON files to text... ---")
    if not os.path.exists(source_dir):
        return

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(source_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(dest_dir, txt_filename)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                text_content = ""
                for entry in data:
                    # Assuming a Q&A format like the provided files
                    if "question" in entry and "answer" in entry:
                        text_content += f"Question: {entry['question']}\nAnswer: {entry['answer']}\n\n"

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_content)

                print(f"Successfully converted '{filename}' to '{txt_filename}'.")
            except Exception as e:
                print(f"Error converting '{filename}': {e}")


if __name__ == "__main__":
    # Define directories relative to the project root
    source_data_dir = "raw_data"
    dest_data_dir = "data/indian_law_docs"

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_data_dir, exist_ok=True)

    convert_pdfs_to_txt(source_data_dir, dest_data_dir)
    convert_json_to_txt(source_data_dir, dest_data_dir)
    print("\nConversion complete! All documents are now in 'data/indian_law_docs/'.")
    print("You can now run 'python data/faiss_indexer.py' to generate the FAISS index.")
