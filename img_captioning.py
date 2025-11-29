import os
import json
import time
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# Configuration (tweak here)
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

INPUT_JSON_PATH = "extract2025-11-29T05-45-36/structuredData.json"
FIGURES_FOLDER = "extract2025-11-29T05-45-36/figures"
OUTPUT_JSON_PATH = os.path.join("output", "image_captions.json")
# Gemini model name
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GEN_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Context extraction (fast)
def get_document_context(elements, current_index, max_chars=200):
    """
    Efficiently extracts nearby text (before/after) around the figure.
    """
    text_before, text_after = [], []

    # Scan BACKWARD
    chars_collected = 0
    for j in range(current_index - 1, -1, -1):
        el = elements[j]
        if "Text" in el and "Figure" not in el.get("Path", ""):
            t = el["Text"].strip()
            if t:
                text_before.insert(0, t)
                chars_collected += len(t)
                if chars_collected >= max_chars:
                    break

    # Scan FORWARD
    chars_collected = 0
    for k in range(current_index + 1, len(elements)):
        el = elements[k]
        if "Text" in el and "Figure" not in el.get("Path", ""):
            t = el["Text"].strip()
            if t:
                text_after.append(t)
                chars_collected += len(t)
                if chars_collected >= max_chars:
                    break

    before_text = " ".join(text_before)[-max_chars:]
    after_text  = " ".join(text_after)[:max_chars]

    return f"{before_text} {after_text}".strip()

# Gemini interaction
def process_image_with_gemini(image_path, elements, index):
    """
    1. Checks image quality
    2. Generates visual + context-aware caption
    3. Returns structured JSON
    """
    doc_context = get_document_context(elements, index)

    try:
        with Image.open(image_path) as img:
            response = GEN_MODEL.generate_content(
                [
                    f"""
                    Analyze this image found in a document.

                    Surrounding text context: "{doc_context}"

                    Return JSON with:
                    - is_useful: boolean
                    - visual_caption: string
                    - final_caption: string
                    """,
                    img
                ],
                generation_config={"response_mime_type": "application/json"}
            )

        # response.text may contain a JSON object or an array; parse safely
        try:
            parsed = json.loads(response.text)
        except Exception:
            # fallback: if response already provides data attribute
            parsed = getattr(response, "data", None) or getattr(response, "content", None) or response

        # If API returned a list, try to pick the first dict-like item
        if isinstance(parsed, list):
            # choose first dict entry, or wrap list into dict under 'items'
            first = None
            for item in parsed:
                if isinstance(item, dict):
                    first = item
                    break
            result = first if first is not None else {"items": parsed}
        elif isinstance(parsed, dict):
            result = parsed
        else:
            # Unexpected type: wrap into dict for downstream defaulting
            result = {"text": str(parsed)}

    except Exception as e:
        print(f" Gemini error for {image_path}: {e}")
        result = {
            "is_useful": False,
            "visual_caption": f"Error: {str(e)}",
            "final_caption": doc_context
        }

    return {
        "image_path": image_path,
        "quality_pass": result.get("is_useful", False),
        "captions": {
            "visual": result.get("visual_caption", ""),
            "context": doc_context,
            "combined": result.get("final_caption", "")
        }
    }

if __name__ == "__main__":
    # Load structured data JSON
    try:
        with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {INPUT_JSON_PATH}: {e}")

    elements = data.get("elements", [])
    figures_folder = FIGURES_FOLDER
    output_json_path = OUTPUT_JSON_PATH

    all_results = []

    # Precompute indices of figure elements
    figure_indices = [
        i for i, el in enumerate(elements)
        if "Figure" in el.get("Path", "")
    ]

    logger.info(f"Found {len(figure_indices)} figure images.")

    for i in figure_indices:
        el = elements[i]
        # try several possible keys for file paths
        file_paths = el.get("filePaths") or el.get("filePath") or el.get("FilePaths") or el.get("FilePath")
        img_file = None

        if isinstance(file_paths, list) and file_paths:
            img_file = file_paths[0]
        elif isinstance(file_paths, str):
            img_file = file_paths

        if not img_file:
            logger.warning(f"No image file listed for element at index {i}")
            continue

        # Normalize path: support values like "figures/name.png", "name.png", or absolute paths
        if os.path.isabs(img_file):
            full_path = img_file
        else:
            # strip any leading ./ or / and join with the figures folder
            img_basename = img_file.lstrip("./\\")
            # If the img_file already contains a directory like "figures/..", use basename
            img_basename = os.path.basename(img_basename)
            full_path = os.path.join(figures_folder, img_basename)

        if not os.path.exists(full_path):
            logger.warning(f"Missing image file: {full_path}")
            continue

        logger.info(f"Processing: {full_path}")

        result = process_image_with_gemini(full_path, elements, i)
        all_results.append(result)

    # Save output once after processing all images
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Finished. Processed {len(all_results)} images.")
    logger.info(f"Saved to: {output_json_path}")

    # Print summary counts
    good = sum(1 for r in all_results if r.get("quality_pass"))
    failed = sum(1 for r in all_results if r.get("error"))
    logger.info(f"Summary: {len(all_results)} processed, {good} passed quality, {failed} failed.")
