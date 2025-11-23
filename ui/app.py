import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.utils.backend import BackendAPI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

backend = BackendAPI()

# Serve images from pics directory
@app.route('/pics/<path:filename>')
def serve_pic(filename):
    # Use more reliable path construction
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pics_dir = os.path.join(base_dir, 'pics')
    return send_from_directory(pics_dir, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file is None:
        return jsonify({'error': 'No selected file'}), 400
    raw_filename = file.filename or ''
    if raw_filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(raw_filename)
    if filename.lower().endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            meta = backend.load_data(filepath)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

        return jsonify({'filepath': filepath, 'metadata': meta})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/build_index', methods=['POST'])
def build_index():
    data = request.get_json()
    filepath = data.get('filepath')
    sparsity = data.get('sparsity')
    field = data.get('field')
    limit = data.get('limit')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400

    try:
        # Reload data if the user switched files without re-uploading
        if backend.data_path != filepath:
            backend.load_data(filepath)
        sparsity_value = float(sparsity) if sparsity not in (None, '') else 0.9
        limit_value = int(limit) if limit not in (None, '') else 1000
        meta = backend.build_index(sparsity_value, field=field, limit=limit_value)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    index_size_kb = meta['index_size_bytes'] / 1024 if meta.get('index_size_bytes') else 0
    return jsonify({
        'message': 'Index built successfully',
        'index_size': f'{index_size_kb:.2f} KB',
        'details': meta
    })

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_str = data.get('query')
    use_index = data.get('use_index', False)
    limit = data.get('limit')

    try:
        limit_value = int(limit) if limit not in (None, '') else 50
        response = backend.query(query_str or '', use_index=bool(use_index), limit=limit_value)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify(response)


@app.route('/analytics', methods=['GET'])
def analytics():
    return jsonify(backend.analyse())

if __name__ == '__main__':
    app.run(debug=True, port=2025, use_reloader=False)
