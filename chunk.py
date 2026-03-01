from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

print("running")

# Paths
RAW_PDF_DIR = Path("data/raw_pdfs")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75
)

# Docling converter
pipeline_options = PdfPipelineOptions(
    do_ocr=False,              # Disable OCR (avoids RapidOCR download)
    do_table_structure=False,  # Disable table models
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

for pdf_path in RAW_PDF_DIR.glob("*.pdf"):
    print(f"Processing {pdf_path.name}")

    try:
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as e:
        print(f"⚠️ Failed to process {pdf_path.name}: {e}")
        continue

    texts = []
    for item, _ in doc.iterate_items():
        if hasattr(item, 'text'):
            text = item.text.strip()
            if len(text) > 50:
                texts.append(text)

    if not texts:
        print(f"⚠️ No usable text in {pdf_path.name}")
        continue

    # Chunk text
    chunks = splitter.split_text("\n".join(texts))

    # Prepare RAG-ready JSON
    chunk_records = []
    for i, chunk in enumerate(chunks):
        chunk_records.append({
            "chunk_id": f"{pdf_path.stem}_{i}",
            "text": chunk,
            "metadata": {"source": pdf_path.name}
        })

    # Save
    output_file = CHUNKS_DIR / f"{pdf_path.stem}_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(chunk_records)} chunks → {output_file.name}")