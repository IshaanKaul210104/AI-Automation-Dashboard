import fitz
import os

pdf_path = os.path.join("uploads", "2025-11-16_17-57-07_SQL GUIDE.pdf")
doc = fitz.open(pdf_path)
print(f"Number of pages: {doc.page_count}")

for i, page in enumerate(doc):
    print(f"--- Page {i+1} ---")
    text = page.get_text()
    if text:
        print(text)
    else:
        print("No text found on this page.")
