# 🔍 Multimodal RAG System with Evaluation Dashboard

A comprehensive multimodal Retrieval-Augmented Generation (RAG) system that combines text, tables, and images for intelligent document querying. Features include PDF extraction using Adobe Extract API, CLIP-based multimodal embeddings, MMR diversity ranking, and an interactive Streamlit evaluation dashboard.

## 📋 Features

### 🎯 Core Capabilities
- **Multimodal Document Processing**: Extract and index text, tables, and images from PDFs
- **Adobe Extract API Integration**: High-quality PDF parsing with structure preservation
- **CLIP Embeddings**: Unified semantic search across text and images
- **MMR Diversity Ranking**: Maximum Marginal Relevance for diverse, comprehensive results
- **Parallel Retrieval**: Async queries across multiple FAISS indexes
- **Smart Reranking**: CLIP-based late fusion with diversity optimization
- **AI-Powered Synthesis**: Gemini-based answer generation with multimodal context

### 📊 Evaluation Dashboard
- **Real-time Metrics**: Precision@K, Recall@K, MRR, NDCG
- **Performance Tracking**: Query latency analysis and component timing
- **Diversity Metrics**: Intra-list diversity and coverage analysis
- **Visual Analytics**: Interactive charts and historical trends
- **Export Functionality**: Download metrics history for analysis

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PDF Documents                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│               Adobe PDF Services API                             │
│                 (ingestion.py)                                   │
│  • ExtractTextInfoFromPDF()                                      │
│  • Reading-order text                                            │
│  • Table structures                                              │
│  • Figure extraction                                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  structuredData.json       │
        │  + figures/ + tables/      │
        └────────────┬───────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  Text Chunking   │    │ Table Cleaning   │
│  (chunking.py)   │    │(clean_tables.py) │
│                  │    │                  │
│ • Page docs      │    │ • Remove _x000D_ │
│ • Semantic chunks│    │ • Normalize data │
└────────┬─────────┘    └────────┬──────────┘
         │                       │
         │              ┌────────▼──────────┐
         │              │ Image Captioning  │
         │              │(img_captioning.py)│
         │              │                   │
         │              │ • Gemini 2.0 Flash│
         │              │ • Visual + Context│
         │              │ • Combined caption│
         │              └────────┬──────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  CLIP Embeddings │    │  CLIP Embeddings │
│   (Text/Table)   │    │     (Images)     │
│                  │    │                  │
│multimodal_indexer│    │multimodal_indexer│
└────────┬─────────┘    └────────┬──────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ FAISS Index      │    │ FAISS Index      │
│  text_tables/    │    │    images/       │
└────────┬─────────┘    └────────┬──────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │  Parallel Retrieval    │
         │  (dual_qa_setup.py)    │
         │                        │
         │  • Async queries       │
         │  • MMR diversity       │
         │  • CLIP reranking      │
         └───────────┬────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │   Gemini Synthesis     │
         │  (Answer Generation)   │
         │                        │
         │  • Multimodal context  │
         │  • Image paths included│
         └────────────────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │  Streamlit Dashboard   │
         │      (app.py)          │
         │                        │
         │  • Interactive UI      │
         │  • Evaluation metrics  │
         │  • Performance tracking│
         └────────────────────────┘
```

---

## 📁 Project Structure

```
multimodal-rag/
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Python dependencies
│
├── ingestion.py             # PDF ingestion pipeline with Adobe Extract API
├── chunking.py              # Text/table chunking strategies
├── img_captioning.py        # Image caption generation
├── clean_tables.py          # Table cleaning and preprocessing
├── multimodal_indexer.py    # FAISS indexing with CLIP embeddings
│
├── dual_qa_setup.py         # Main RAG system with MMR
├── evaluation_metrics.py    # Comprehensive metrics library
├── app.py                   # Streamlit dashboard
│
└── faiss_indexes/           # Generated FAISS indexes
    ├── text_tables/         # Text and table embeddings
    └── images/              # Image embeddings
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Adobe PDF Services API credentials
- Google Gemini API key
- 8GB+ RAM recommended
- (Optional) CUDA GPU for faster processing

### 2. Installation

```bash
# Clone the repository
git clone [<(https://github.com/Khushi-c-sharma/multimodal-rag-agent)>]
cd multimodal-rag-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Adobe PDF Services API
ADOBE_CLIENT_ID=your_adobe_client_id
ADOBE_CLIENT_SECRET=your_adobe_client_secret

# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key

# Optional: Paths
PDF_INPUT_DIR=./data/pdfs
OUTPUT_DIR=./data/extracted
FAISS_INDEX_DIR=./faiss_indexes
```

**Get API Keys:**
- Adobe PDF Services: https://developer.adobe.com/document-services/apis/pdf-extract/
- Google Gemini: https://makersuite.google.com/app/apikey

### 4. Prepare Your Data

```bash
# Create data directories
mkdir -p data/pdfs data/extracted

# Place your PDF files in data/pdfs/
cp your_documents.pdf data/pdfs/
```

### 5. Run the Pipeline

#### Complete Pipeline Flow:

```bash
# Step 1: Extract from PDFs using Adobe API
python -c "from ingestion import ExtractTextInfoFromPDF; ExtractTextInfoFromPDF()"

# Step 2: Process extracted data
# This will create structuredData.json and extract figures/tables
# Output: output/ExtractTextInfoFromPDF/extract<timestamp>.zip

# Step 3: Unzip the extraction
unzip output/ExtractTextInfoFromPDF/extract*.zip -d data/extracted/

# Step 4: Run chunking pipeline
python -c "from chunking import run_full_chunking_pipeline; \
run_full_chunking_pipeline('data/extracted/structuredData.json', 'data/output')"

# Step 5: Generate image captions
python img_captioning.py

# Step 6: Clean tables
python -c "from clean_tables import load_clean_and_save_tables; \
load_clean_and_save_tables('./data/extracted/tables', './data/clean/tables_csv', './data/clean/tables_xlsx')"

# Step 7: Create FAISS indexes
python multimodal_indexer.py

# Step 8: Launch dashboard
streamlit run app.py
```

#### Quick Start (Automated):

Or use the complete pipeline script:

```bash
# Run entire pipeline
python run_pipeline.py --pdf_path data/pdfs/document.pdf
```

---

## 📚 Module Documentation

### Core Pipeline Modules

#### 🔹 1. PDF Extraction — `ingestion.py`

High-fidelity extraction powered by **Adobe PDF Services API**.

**Provides:**
- ✅ Reading-order text extraction
- ✅ Table structures (with cell-level data)
- ✅ Figure images (high-quality renditions)
- ✅ `structuredData.json` + renditions ZIP output

**Key Function:**
```python
from ingestion import ExtractTextInfoFromPDF

# Extract from PDF
ExtractTextInfoFromPDF()
```

**Output Structure:**
```
output/ExtractTextInfoFromPDF/
└── extractYYYY-MM-DDTHH-MM-SS.zip
    ├── structuredData.json      # Main extraction data
    ├── figures/
    │   ├── fig_001.png
    │   ├── fig_002.png
    │   └── ...
    └── tables/
        ├── table_001.xlsx
        ├── table_002.xlsx
        └── ...
```

**Adobe API Features:**
- Maintains reading order
- Preserves document structure
- Extracts high-resolution images
- Handles complex multi-column layouts
- Identifies tables with cell boundaries

---

#### 🔹 2. Text Chunking — `chunking.py`

Transforms `structuredData.json` into processable chunks.

**Output Types:**
1. **Raw Text**: Reading-order text from entire document
2. **Page Documents**: Page-level Document objects
3. **Chunked Documents**: Semantic chunks for RAG

**Usage:**
```python
from chunking import run_full_chunking_pipeline

run_full_chunking_pipeline(
    input_json="path/to/structuredData.json",
    output_folder="data/output"
)
```

**Outputs:**
```
data/output/
├── extracted_text.txt      # Full document text
├── page_docs.json          # Page-level chunks
└── chunked_docs.json       # Semantic chunks
```

**Chunking Strategy:**
- Respects paragraph boundaries
- Maintains context windows
- Preserves table integrity
- Configurable chunk size and overlap

---

#### 🔹 3. Figure Captioning — `img_captioning.py`

Generates **multimodal, context-aware** captions using **Gemini 2.0 Flash**.

**Features:**
- 🎨 Visual description from image analysis
- 📝 Context extraction from nearby text
- 🔗 Combined caption with full context
- ✅ Quality validation

**Usage:**
```python
python img_captioning.py
```

**Output:**
```
output/image_captions.json
```

**Example Output:**
```json
{
  "image_path": "figures/fig_001.png",
  "quality_pass": true,
  "captions": {
    "visual": "Bar chart showing GDP growth from 2015-2024, with increasing trend",
    "context": "Figure 2.1 from Section 2: Economic Indicators. Referenced in paragraph discussing macroeconomic trends.",
    "combined": "Bar chart showing GDP growth from 2015-2024, with increasing trend. Figure 2.1 from Section 2: Economic Indicators, illustrating macroeconomic trends in Qatar's economy."
  }
}
```

**Caption Types:**
- **Visual**: Pure image description
- **Context**: Surrounding text context
- **Combined**: Unified multimodal caption (used for indexing)

---

#### 🔹 4. Table Cleaning — `clean_tables.py`

Normalizes extracted Excel tables from Adobe API.

**Cleaning Operations:**
- Removes `_x000D_` artifacts
- Fixes whitespace and formatting
- Standardizes cell values
- Saves as clean CSV/XLSX

**Usage:**
```python
from clean_tables import load_clean_and_save_tables

load_clean_and_save_tables(
    folder_path="./extracted/tables",
    output_csv_folder="./clean/tables_csv",
    output_xlsx_folder="./clean/tables_xlsx"
)
```

**Cleaning Steps:**
1. Remove special characters (`_x000D_`, etc.)
2. Strip extra whitespace
3. Normalize cell formatting
4. Validate table structure
5. Save in multiple formats

**Output:**
```
clean/
├── tables_csv/
│   ├── table_001.csv
│   └── table_002.csv
└── tables_xlsx/
    ├── table_001.xlsx
    └── table_002.xlsx
```

#### `multimodal_indexer.py`
FAISS index creation with CLIP embeddings.

**Key Classes:**
- `CLIPEmbedding`: CLIP embedding wrapper for LangChain
- `MultimodalIndexer`: Creates FAISS indexes

**Usage:**
```python
from multimodal_indexer import MultimodalIndexer

indexer = MultimodalIndexer(
    clip_model="clip-ViT-B-32"
)

# Create indexes
indexer.create_text_index(chunks, save_path="./faiss_indexes/text_tables")
indexer.create_image_index(images, save_path="./faiss_indexes/images")
```

### RAG System Modules

#### `dual_qa_setup.py`
Main RAG system with optimized retrieval.

**Key Classes:**
- `ParallelRAGClipOnly`: Main agent with async retrieval
- `CLIPReranker`: MMR-based reranking with diversity

**Key Features:**
- Parallel retrieval from multiple indexes
- MMR at retrieval and reranking stages
- Direct embedding extraction from FAISS
- Configurable diversity parameters

**Usage:**
```python
from dual_qa_setup import setup_system

agent = setup_system(
    text_tables_path="./faiss_indexes/text_tables",
    images_path="./faiss_indexes/images",
    top_k=10,
    lambda_mult=0.5  # Diversity parameter
)

result = agent.ask("What are Qatar's GDP trends?")
print(result['answer'])
```

#### `evaluation_metrics.py`
Comprehensive metrics for RAG evaluation.

**Key Functions:**
- `calculate_precision_at_k()`: Precision@K
- `calculate_recall_at_k()`: Recall@K
- `calculate_mrr()`: Mean Reciprocal Rank
- `calculate_ndcg()`: Normalized DCG
- `calculate_diversity_score()`: Result diversity
- `calculate_intra_list_diversity()`: ILD metric

**Usage:**
```python
from evaluation_metrics import calculate_retrieval_metrics

metrics = calculate_retrieval_metrics(
    retrieved_items=results,
    relevant_ids=ground_truth,
    k_values=[1, 3, 5, 10]
)
```

#### `app.py`
Interactive Streamlit dashboard.

**Features:**
- Three-tab interface (Query, Metrics, Analytics)
- Real-time performance monitoring
- Historical trend analysis
- Configurable parameters
- CSV export

---

## ⚙️ Configuration

### Diversity Control (λ Parameters)

The system uses lambda (λ) parameters to control diversity vs relevance:

**λ = 1.0**: Pure relevance (no diversity optimization)
**λ = 0.7**: Slight preference for relevance
**λ = 0.5**: Balanced relevance and diversity
**λ = 0.3**: High diversity
**λ = 0.0**: Maximum diversity (pure dissimilarity)

### Recommended Settings

**Research/Exploration Mode:**
```python
retrieval_k=20
fetch_k=50
lambda_mult=0.3      # High diversity in retrieval
rerank_lambda=0.5    # Balanced reranking
```

**Production/Precision Mode:**
```python
retrieval_k=10
fetch_k=30
lambda_mult=0.9      # High relevance in retrieval
rerank_lambda=0.9    # High relevance in reranking
```

**Balanced Mode (Default):**
```python
retrieval_k=15
fetch_k=40
lambda_mult=0.5
rerank_lambda=0.7
```

### CLIP Model Selection

- `clip-ViT-B-32`: Faster, 512-dim embeddings, good for most tasks
- `clip-ViT-L-14`: Slower, 768-dim embeddings, higher accuracy

---

## 📊 Dashboard Usage

### Tab 1: Query Interface
1. Enter your question or select a sample query
2. Adjust parameters in sidebar (optional)
3. Click "🚀 Search"
4. View:
   - AI-generated comprehensive answer
   - Retrieved text chunks with scores
   - Retrieved images with captions and paths
   - Source metadata

### Tab 2: Metrics Dashboard
Real-time metrics for the latest query:
- ⏱️ **Latency**: End-to-end processing time
- 📊 **Score Distribution**: Min, max, average relevance
- 🌈 **Diversity Score**: Result variety measure
- 📈 **Type Distribution**: Text vs image balance
- 🎯 **Ranking Metrics**: MRR, NDCG (if ground truth available)

### Tab 3: Analytics
Historical performance analysis:
- Latency trends over time
- Score distribution box plots
- Query history table
- Export to CSV for external analysis
- Performance statistics (mean, std, percentiles)

---

## 🐛 Troubleshooting

### Adobe API Issues

**Error: "Invalid credentials"**
```bash
# Verify your credentials
echo $ADOBE_CLIENT_ID
echo $ADOBE_CLIENT_SECRET

# Check .env file
cat .env | grep ADOBE
```

**Solutions:**
- Verify `ADOBE_CLIENT_ID` and `ADOBE_CLIENT_SECRET` in `.env`
- Check credentials at [Adobe Developer Console](https://developer.adobe.com/console)
- Ensure PDF Services API is enabled for your project
- Regenerate credentials if expired

**Error: "API quota exceeded"**
- Check your Adobe API usage limits in the console
- Free tier: 500 API calls per month
- Consider upgrading plan for production use
- Implement request batching and caching

**Error: "File too large"**
- Adobe API limit: 100 MB per PDF
- Split large PDFs into smaller files
- Use compression tools before processing

**Error: "Unsupported PDF version"**
- Adobe supports PDF 1.3 to 2.0
- Use Adobe Acrobat to convert older PDFs
- Check PDF integrity with `pdfinfo` command

---

### Memory Issues

**OutOfMemoryError during indexing:**
- Process PDFs in smaller batches
- Reduce `chunk_size` in chunking
- Use `faiss-cpu` instead of loading all to GPU
- Increase system swap space

### Slow Performance

**First query is slow:**
- Normal - models are loading
- Subsequent queries will be faster due to caching

**All queries are slow:**
- Use GPU version: `pip install faiss-gpu`
- Reduce `fetch_k` and `retrieval_k`
- Use smaller CLIP model (ViT-B-32)
- Enable model caching

---

## 📈 Performance Optimization

### Indexing Optimization
```python
# Batch processing for large datasets
indexer.create_text_index(
    chunks,
    batch_size=100,  # Process in batches
    use_gpu=True     # If available
)
```

### Query Optimization
```python
agent = setup_system(
    top_k=5,           # Fewer results = faster
    retrieval_k=10,    # Smaller pool
    fetch_k=20,        # Smaller MMR pool
    lambda_mult=0.9    # Less MMR computation
)
```

### Caching Strategy
The system caches:
- ✅ FAISS indexes (at startup)
- ✅ CLIP model (singleton pattern)
- ✅ Query embeddings (LRU cache)
- ✅ Streamlit resources (@st.cache_resource)

---

## 🧪 Testing

### Unit Tests
```bash
# Test individual modules
python -m pytest tests/test_chunking.py
python -m pytest tests/test_metrics.py
```

### Integration Test
```bash
# Test full pipeline
python test_pipeline.py
```

### Evaluation Dataset
Create `eval_dataset.json`:
```json
{
  "queries": [
    {
      "query": "What is Qatar's GDP?",
      "relevant_docs": ["doc1", "doc2"]
    }
  ]
}
```

Run evaluation:
```bash
python evaluate_system.py --dataset eval_dataset.json
```

---

## 🔐 Security Best Practices

1. **Never commit API keys**: Use `.env` and `.gitignore`
2. **Validate inputs**: Sanitize user queries
3. **Rate limiting**: Implement for production
4. **Access control**: Restrict dashboard in production
5. **Data privacy**: Ensure PDFs don't contain sensitive info

---

## 📦 Deployment

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t multimodal-rag .
docker run -p 8501:8501 --env-file .env multimodal-rag
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **AWS**: EC2 + ECS or Lambda
- **GCP**: Cloud Run or Compute Engine
- **Azure**: Container Instances

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
```

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Adobe PDF Services**: High-quality PDF extraction
- **OpenAI CLIP**: Multimodal embeddings
- **FAISS**: Efficient similarity search
- **LangChain**: RAG framework
- **Streamlit**: Interactive dashboards
- **Google Gemini**: Answer synthesis

---

## 🗺️ Roadmap

- [ ] Support for additional document formats (Word, Excel)
- [ ] Multi-language support
- [ ] Advanced table understanding with TableTransformer
- [ ] Graph-based retrieval
- [ ] Fine-tuned reranking models
- [ ] Real-time collaborative queries
- [ ] API endpoint deployment
- [ ] Benchmark suite with standard datasets

---

**Last Updated**: November 2025
**Version**: 1.0.0

---

*Built with ❤️ for advanced document intelligence and multimodal information retrieval*
