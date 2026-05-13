# P2N: Plot to Numbers

P2N (Plot 2 Numbers) is an end-to-end framework for parsing complex scientific plots directly into structured numerical data (JSON). It leverages the vision-encoder-decoder architecture (Donut) to achieve OCR-free chart derendering.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data:**
   The `generate_data.py` script creates synthetic charts and corresponding ground truth metadata for training. It supports a huge variety of scientific plot types including scatter, bar, pie, histogram, density, contour, corner plots, phase diagrams, Ashby charts, parity grids, stress-strain curves, and high-energy physics limit/bump-hunt plots.
   ```bash
   python generate_data.py --samples 1000 --output_dir data
   ```

3. **Train the Model:**
   Train the Donut model on your generated dataset using CUDA.
   ```bash
   python train.py
   ```

4. **Evaluate:**
   (Evaluation scripts pending implementation for numerical accuracy metrics like MAE/RMSE).
