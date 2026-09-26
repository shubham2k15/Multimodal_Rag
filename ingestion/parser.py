"""
Parses a PDF into structured elements using Unstructured.io.
Tables come back with HTML structure preserved. Images/figures are extracted
to disk as real image files, referenced by path.
"""
from unstructured.partition.pdf import partition_pdf

from config import IMAGE_STORE_DIR


def parse_pdf(pdf_path: str):
    """
    Returns an ordered list of Unstructured elements: Title, NarrativeText,
    Table, Image, Figure, FigureCaption — each carrying page_number metadata.
    """
    print(f"Parsing PDF with Unstructured.io: {pdf_path}")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",                  # required for table/figure region detection
        infer_table_structure=True,         # gives element.metadata.text_as_html for tables
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Figure"],
        extract_image_block_output_dir=IMAGE_STORE_DIR,
    )
    print(f"Parsed {len(elements)} elements from {pdf_path}")
    return elements
