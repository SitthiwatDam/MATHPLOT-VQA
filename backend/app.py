from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=['OPTIONS', 'POST'])

UPLOAD_FOLDER = './uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

def prompt_formatting(summary, question):
    return f"""
Below is a summary that describes a chart, paired with a question the user has about the chart. Write a response that appropriately answer the question.

### Summary:
{summary}

### Question:
{question}

### Response:
""".strip()

chart_sum_prompt = "What is this chart about?"

# Matcha
processor = Pix2StructProcessor.from_pretrained('google/matcha-chart2text-pew')
chart_summarizer = Pix2StructForConditionalGeneration.from_pretrained('google/matcha-chart2text-pew').to(device)

# MiniChat
tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B", use_fast=False)
qa_model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()

MAX_PATCHES = 1024

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        print('No file part', flush=True)
        return jsonify({'error': 'No file part'}), 400

    file = request.files.get('image')  # Get the file object
    if file.filename == '':
        print('No selected file', flush=True)
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Chart summarization
    image = Image.open(file_path).convert("RGB")
    print("Image input obtained", flush=True)
    inputs = processor(images=image, text=chart_sum_prompt, return_tensors="pt",  add_special_tokens=True, max_patches=MAX_PATCHES).to(device)
    predictions = chart_summarizer.generate(**inputs, max_new_tokens=256)
    summary = processor.decode(predictions[0], skip_special_tokens=True)
    print("Summary input obtained", flush=True)

    # QA model process
    query = request.form['text']
    print("Query input obtained", flush=True)
    prompt = prompt_formatting(summary, query)
    # conv.append_message(conv.roles[0], user_input)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids
    output_ids = qa_model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=256,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print("Answer output obtained", flush=True)

    return jsonify({'message': 'Output generated successfully', 'text-generated': output}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
