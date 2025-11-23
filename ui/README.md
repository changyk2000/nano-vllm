# Nano-vLLM Web UI

## Overview
Interactive web interface for Nano-vLLM with KV cache indexing and querying capabilities.

## Features
- 📁 CSV data upload and analysis
- ⚡ KV cache index building with configurable sparsity
- 🔍 Hybrid query execution (with/without index)
- 📊 Real-time analytics with auto-generated charts
- 🎨 Modern, responsive UI design

## Running the Application

1. **Install dependencies:**
   ```bash
   pip install -r ui/requirements.txt
   ```

2. **Start the server:**
   ```bash
   python ui/app.py
   ```

3. **Access the UI:**
   Open your browser to `http://localhost:2025`

## Architecture

### Backend (`nanovllm/utils/backend.py`)
The backend API handles:
- Data loading and inference
- Index building with KV cache pruning
- Query execution with LLM filtering
- **Analytics and plotting** (server-side chart generation)

### Frontend
- **Templates** (`ui/templates/`): HTML structure with modern design
- **Static Assets** (`ui/static/`):
  - `style.css`: Modern gradient theme with responsive design
  - `script.js`: Client-side logic for UI interactions
  - `pics/`: Auto-generated chart images (created at runtime)

### Analytics Charts
Charts are now generated on the backend using matplotlib and saved as PNG images:
1. **Latency Comparison**: Bar chart comparing indexed vs non-indexed query performance
2. **Result Drift**: Scatter plot showing accuracy differences between methods
3. **Sparsity Timeline**: Line chart tracking index sparsity over time
4. **KV Latency Timeline**: Dual-line chart for transfer and compute times

## Recent Changes

### Backend Plotting Migration
- Moved chart generation from client-side D3.js to server-side matplotlib
- Charts are generated during analytics phase and saved to `/pics` directory
- Improved performance by eliminating client-side rendering overhead
- Better caching and consistent rendering across browsers

### UI Beautification
- Modern purple gradient color scheme
- Enhanced card animations and hover effects
- Numbered step indicators with gradient badges
- Improved form inputs and button styles
- Better responsive design for mobile devices
- All text converted to English

## Configuration

The server runs on port **2025** by default. To change:
```python
# In ui/app.py
app.run(debug=True, port=YOUR_PORT, use_reloader=False)
```

## Notes
- SSH port forwarding may be required to access the UI remotely
- The backend API is defined in `nanovllm/utils/backend.py`
- Generated chart images are automatically excluded from git (.gitignore)
