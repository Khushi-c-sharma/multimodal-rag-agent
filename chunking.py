import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1 — Extract raw text (reading order)
def extract_text_from_json(data):
    elements = data.get("elements", [])
    pages = {}

    for el in elements:
        if "Text" in el and el["Text"].strip():
            pages.setdefault(el.get("Page", 0), []).append(el)

    all_text = []

    for page_num in sorted(pages.keys()):
        page_elements = pages[page_num]

        page_elements.sort(
            key=lambda x: (-x.get("Bounds", [0,0,0,0])[1],
                           x.get("Bounds", [0,0,0,0])[0])
        )

        page_text = [
            el["Text"].strip() for el in page_elements if el["Text"].strip()
        ]

        all_text.extend(page_text)

    return "\n".join(all_text)


# STEP 2 — Create Document objects w/ metadata
def extract_docs_with_metadata(data):
    elements = data.get("elements", [])
    pdf_metadata = data.get("extended_metadata", {})

    pages = {}

    for el in elements:
        if "Text" in el and el["Text"].strip():
            pages.setdefault(el.get("Page", 0), []).append(el)

    docs = []

    for page_num in sorted(pages.keys()):
        page_elements = pages[page_num]

        page_elements.sort(
            key=lambda x: (-x.get("Bounds", [0,0,0,0])[1],
                           x.get("Bounds", [0,0,0,0])[0])
        )

        page_text_list = []
        metadata_list = []

        for el in page_elements:
            if el["Text"].strip():
                page_text_list.append(el["Text"].strip())
                metadata_list.append({
                    "bounds": el.get("Bounds"),
                    "object_id": el.get("ObjectID"),
                    "path": el.get("Path"),
                    "font": el.get("Font"),
                    "lang": el.get("Lang"),
                })

        page_text = "\n".join(page_text_list)

        docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "page": page_num,
                    "pdf_meta": pdf_metadata,
                    "elements": metadata_list
                }
            )
        )

    return docs


# STEP 3 — Chunk docs
def split_docs_into_chunks(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return chunked_docs


# MAIN PIPELINE
def run_full_chunking_pipeline(input_json, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Load JSON
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # STEP 1
    final_text = extract_text_from_json(data)
    with open(os.path.join(output_folder, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(final_text)

    # STEP 2
    page_docs = extract_docs_with_metadata(data)

    # STEP 3
    chunked_docs = split_docs_into_chunks(page_docs)

    # Save intermediate JSON (preserve unicode)
    with open(os.path.join(output_folder, "page_docs.json"), "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in page_docs], f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_folder, "chunked_docs.json"), "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in chunked_docs], f, indent=2, ensure_ascii=False)

    print("Chunking pipeline complete.")
    print(f"Pages: {len(page_docs)} | Chunks: {len(chunked_docs)}")

    return chunked_docs


if __name__ == "__main__":
    input_json_path = "extract2025-11-29T05-45-36/structuredData.json"
    output_folder = os.path.join("data", "output")
    os.makedirs(output_folder, exist_ok=True)

    run_full_chunking_pipeline(
        input_json=input_json_path,
        output_folder=output_folder
    )
