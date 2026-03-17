from src.pipeline import analyze_document

image_path = "data/samples/receipt-3.jpg"

result = analyze_document(image_path)

print(result)