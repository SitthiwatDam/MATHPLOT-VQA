from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=['OPTIONS', 'POST'])

UPLOAD_FOLDER = './uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        print('No file part')
        return jsonify({'error': 'No file part'}), 400

    file = request.files.get('image')  # Get the file object
    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return jsonify({'message': 'File uploaded successfully', 'text-generated': "This is an example of texts generated"}), 200

if __name__ == '__main__':
    app.run(debug=True)
