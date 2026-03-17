import time

from src.qualitycheck import check_image_quality
from src.preprocess import preprocess_image
from src.ocr import extract_text
from src.parser import parse_document

def analyze_document(image_path):
    try:
        start = time.time()
        quality = check_image_quality(image_path)
        processed = preprocess_image(image_path)

        text = extract_text(processed)

        parsed = parse_document(text)
        end = time.time()

        return {
            "ocr_text": text,
            "parsed": parsed,
            "processing_time": end - start
        }

    except Exception as e:

        return {
            "error": str(e)
        }