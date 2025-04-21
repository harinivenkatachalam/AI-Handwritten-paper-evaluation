from flask import Flask, request, render_template# type: ignore
import os
import cv2# type: ignore
import easyocr # type: ignore
import torch# type: ignore
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from rapidfuzz import fuzz# type: ignore

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load models
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
reader = easyocr.Reader(['en'])

def detect_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb)
    extracted_texts = []

    for (bbox, _, _) in results:
        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])
        cropped_region = image_rgb[y1:y2, x1:x2]
        pil_image = Image.fromarray(cropped_region)
        inputs = processor(pil_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            output = model.generate(inputs)
        recognized_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        extracted_texts.append(recognized_text)

    return " ".join(extracted_texts).strip()

def compare_texts(key_text, test_text):
    return fuzz.ratio(key_text.lower(), test_text.lower())

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        answer_key = request.files.get("answer_key")
        exam_paper = request.files.get("exam_paper")

        if not answer_key or not exam_paper:
            return render_template("index.html", error="Please upload both files.")

        key_path = os.path.join(UPLOAD_DIR, answer_key.filename)
        exam_path = os.path.join(UPLOAD_DIR, exam_paper.filename)
        answer_key.save(key_path)
        exam_paper.save(exam_path)

        try:
            with open(key_path, "r", encoding="utf-8") as f:
                key_text = f.read().strip()
        except:
            return render_template("index.html", error="Could not read answer key.")

        test_text = detect_text(exam_path)
        if not test_text:
            return render_template("index.html", error="Could not extract text from exam paper.")

        score = compare_texts(key_text, test_text)
        total_marks = 10
        obtained_marks = (score / 100) * total_marks 

        return render_template("index.html", key_text=key_text, test_text=test_text,
                               score=f"{score}%", obtained_marks=f"{obtained_marks:.2f}",
                               total_marks=total_marks)

    return render_template("index.html")
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

