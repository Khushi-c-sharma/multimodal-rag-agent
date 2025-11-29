import os
import json
import logging
import pandas as pd

from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ============================================================
# EMBEDDINGS: CLIP TEXT ENCODER WRAPPER
# ============================================================

class CLIPEmbedding:
    """
    Wrapper around SentenceTransformer CLIP model so that FAISS
    can call embed_documents() or embed_query().
    """

    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)



# ============================================================
# CLASS-BASED MULTIMODAL INDEXER
# ============================================================

class MultimodalIndexer:
    """
    Class-based multimodal FAISS indexing pipeline.
    Supports:
        - text chunks
        - tables (CSV → text)
        - image caption documents

    Embedding model: CLIP (SentenceTransformer)
    """

    def __init__(self,
                 faiss_output_dir: str,
                 model_name: str = "clip-ViT-B-32",
                 batch_size: int = 32):

        self.output_dir = faiss_output_dir
        self.batch_size = batch_size
        self.embedder = CLIPEmbedding(model_name)

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # LOADING FUNCTIONS
    # --------------------------------------------------------

    def load_text_chunks(self, chunked_docs: List[Document]) -> List[Dict]:
        docs = []

        for i, d in enumerate(chunked_docs):
            text = d.page_content.strip()
            if not text:
                continue

            docs.append({
                "id": f"text_{i}",
                "type": "text",
                "content": text,
                "metadata": d.metadata
            })

        logger.info(f"Loaded {len(docs)} text docs")
        return docs

    def load_tables(self, tables_folder: str) -> List[Dict]:
        if not tables_folder or not os.path.exists(tables_folder):
            logger.warning(f"Table folder not found: {tables_folder}")
            return []

        docs = []

        for file in os.listdir(tables_folder):
            if not file.lower().endswith(".csv"):
                continue

            path = os.path.join(tables_folder, file)
            try:
                df = pd.read_csv(path)
            except:
                df = pd.read_csv(path, encoding="latin-1")

            if df.empty:
                continue

            docs.append({
                "id": file[:-4],
                "type": "table",
                "content": df.to_string(index=False),
                "metadata": {
                    "source": "table",
                    "filename": file,
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                }
            })

        logger.info(f"Loaded {len(docs)} tables")
        return docs

    def load_image_captions(self, captions_json_path: str) -> List[Dict]:
        if not captions_json_path or not os.path.exists(captions_json_path):
            logger.warning(f"No captions JSON found: {captions_json_path}")
            return []

        with open(captions_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        docs = []
        for entry in data:
            if not entry.get("quality_pass", False):
                continue

            captions = entry.get("captions", {})
            caption_text = captions.get("combined") or captions.get("visual")
            if not caption_text:
                continue

            img_path = entry.get("image_path", "")
            fig_id = os.path.splitext(os.path.basename(img_path))[0]

            docs.append({
                "id": fig_id,
                "type": "image",
                "content": caption_text,
                "metadata": {
                    "source": "figure",
                    "image_path": img_path,
                    "figure_id": fig_id
                }
            })

        logger.info(f"Loaded {len(docs)} image caption docs")
        return docs

    # --------------------------------------------------------
    # DATA COMBINER
    # --------------------------------------------------------

    def build_dataset(self, text_chunks, captions_json_path, tables_folder):
        text_docs = self.load_text_chunks(text_chunks)
        image_docs = self.load_image_captions(captions_json_path)
        table_docs = self.load_tables(tables_folder)

        all_docs = text_docs + image_docs + table_docs

        logger.info(
            f"Dataset summary:"
            f"\n  Text:   {len(text_docs)}"
            f"\n  Images: {len(image_docs)}"
            f"\n  Tables: {len(table_docs)}"
            f"\n  TOTAL:  {len(all_docs)}"
        )

        return text_docs, image_docs, table_docs

    # --------------------------------------------------------
    # FAISS BUILDERS
    # --------------------------------------------------------

    def _build_faiss(self, docs: List[Dict], save_path: str) -> FAISS:
        if not docs:
            logger.warning(f"No documents to index for {save_path}")
            return None

        texts = [d["content"] for d in docs]
        meta = [{**d["metadata"], "id": d["id"], "type": d["type"]} for d in docs]

        total = len(texts)
        batches = (total + self.batch_size - 1) // self.batch_size

        vectorstore = None

        for b in range(batches):
            s, e = b * self.batch_size, min((b + 1) * self.batch_size, total)
            batch_texts = texts[s:e]
            batch_metas = meta[s:e]

            if vectorstore is None:
                vectorstore = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=self.embedder,
                    metadatas=batch_metas
                )
            else:
                vectorstore.add_texts(batch_texts, batch_metas)

        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        logger.info(f"Saved FAISS index → {save_path}")
        return vectorstore

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    def _load_faiss(self, save_path: str):
        if not os.path.exists(save_path):
            return None

        try:
            vs = FAISS.load_local(
                save_path,
                self.embedder,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing index from {save_path}")
            return vs
        except:
            return None

    # --------------------------------------------------------
    # HIGH LEVEL PIPELINE (AUTO-SKIP REDUNDANCY)
    # --------------------------------------------------------

    def build_separate_indexes(self, text_chunks, captions_json_path, tables_folder):
        """
        Builds:
            /output_dir/text_tables
            /output_dir/images

        Auto-skips if index already exists.
        """

        text_docs, image_docs, table_docs = self.build_dataset(
            text_chunks,
            captions_json_path,
            tables_folder
        )

        tt_path = os.path.join(self.output_dir, "text_tables")
        img_path = os.path.join(self.output_dir, "images")

        vectorstores = {}

        # ----------------------------------------------------
        # TEXT + TABLES
        # ----------------------------------------------------
        existing_tt = self._load_faiss(tt_path)
        if existing_tt:
            logger.info("✓ Text+Tables index already exists — skipping")
            vectorstores["text_tables"] = existing_tt
        else:
            logger.info("Building new Text+Tables index…")
            vectorstores["text_tables"] = self._build_faiss(
                docs=text_docs + table_docs,
                save_path=tt_path
            )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------
        existing_img = self._load_faiss(img_path)
        if existing_img:
            logger.info("✓ Image index already exists — skipping")
            vectorstores["images"] = existing_img
        else:
            logger.info("Building new Images index…")
            vectorstores["images"] = self._build_faiss(
                docs=image_docs,
                save_path=img_path
            )

        return vectorstores

# ============================================================
# MAIN FUNCTION (Runnable CLI-style entrypoint)
# ============================================================

def load_chunked_docs(path: str):
    """Load LangChain Document objects from JSON or Pickle."""
    if not path or not os.path.exists(path):
        raise ValueError(f"Text chunks path not found: {path}")

    logger.info(f"Loading text chunks from: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        docs = []
        for obj in raw:
            if isinstance(obj, dict):
                content = obj.get("page_content") or obj.get("content", "")
                metadata = obj.get("metadata", {})
                if content:
                    docs.append(Document(page_content=content, metadata=metadata))
            elif isinstance(obj, str):
                docs.append(Document(page_content=obj, metadata={}))

        logger.info(f"Loaded {len(docs)} chunks from JSON")
        return docs

    elif path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            docs = pickle.load(f)
        logger.info(f"Loaded {len(docs)} chunks from pickle")
        return docs

    else:
        raise ValueError("Unsupported text chunks format. Use .json or .pkl")


def main(text_chunks_path: str,
         captions_json_path: str,
         tables_folder: str,
         output_dir: str = "faiss_indexes",
         model_name: str = "clip-ViT-B-32",
         batch_size: int = 32,
         mode: str = "both"):
    """
    Run the multimodal index builder.

    Args:
        text_chunks_path: Path to JSON/PKL text chunks
        captions_json_path: Caption JSON for images
        tables_folder: Folder containing CSV tables
        output_dir: Where FAISS indexes will be stored
        model_name: CLIP model name
        batch_size: Embedding batch size
        mode: 'both', 'text_tables', or 'images'
    """

    logger.info("\n" + "=" * 70)
    logger.info("MULTIMODAL INDEXING PIPELINE STARTED")
    logger.info("=" * 70)

    # ---------- Load text chunks ----------
    chunked_docs = load_chunked_docs(text_chunks_path)

    # ---------- Initialize Indexer ----------
    indexer = MultimodalIndexer(
        faiss_output_dir=output_dir,
        model_name=model_name,
        batch_size=batch_size
    )

    # ---------- Build Dataset ----------
    text_docs, image_docs, table_docs = indexer.build_dataset(
        text_chunks=chunked_docs,
        captions_json_path=captions_json_path,
        tables_folder=tables_folder
    )

    # ---------- Run Selected Mode ----------
    vectorstores = {}

    if mode in ("both", "text_tables"):
        logger.info("\n--- Building TEXT + TABLES index ---")
        vectorstores.update(
            {"text_tables": indexer._build_faiss(
                docs=text_docs + table_docs,
                save_path=os.path.join(output_dir, "text_tables")
            )}
        )

    if mode in ("both", "images"):
        logger.info("\n--- Building IMAGES index ---")
        vectorstores.update(
            {"images": indexer._build_faiss(
                docs=image_docs,
                save_path=os.path.join(output_dir, "images")
            )}
        )

    logger.info("\n" + "=" * 70)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 70 + "\n")

    return vectorstores

if __name__ == "__main__":
    CHUNKED_DOCS = "C:\\Users\\khush\\Desktop\\iAI Solutions Assignment\\data\\output\\chunked_docs.json"
    IMAGE_CAPTIONS = "C:\\Users\\khush\\Desktop\\iAI Solutions Assignment\\data\\output\\image_captions.json"
    TABLES_FOLDER = "C:\\Users\\khush\\Desktop\\iAI Solutions Assignment\\data\\output\\cleaned_tables_csv"
    OUTPUT_DIR = "faiss_indexes"

    main(
        text_chunks_path=CHUNKED_DOCS,
        captions_json_path=IMAGE_CAPTIONS,
        tables_folder=TABLES_FOLDER,
        output_dir=OUTPUT_DIR,
        model_name="clip-ViT-B-32",
        batch_size=32,
        mode="both"      # options: both / text_tables / images
    )



