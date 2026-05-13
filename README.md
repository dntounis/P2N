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
   # Generate a large dataset with auxiliary tasks and 30% degraded images
   python generate_data.py --samples 50000 --output_dir data --aux_tasks --degrade_fraction 0.3
   ```

3. **Train the Model (Multi-GPU cluster):**
   The `train.py` script uses Hugging Face's `Seq2SeqTrainer` and supports Distributed Data Parallel (DDP). Use `torchrun` to train across multiple GPUs (e.g., 4x A100s).
   ```bash
   torchrun --nproc_per_node=4 train.py \
       --dataset_path data \
       --output_dir ./p2n-model \
       --epochs 30 \
       --batch_size 4 \
       --gradient_accumulation_steps 8 \
       --max_length 1536
   ```

4. **Evaluate and Infer:**
   The `evaluate.py` and `inference.py` scripts automatically detect and utilize all available GPUs using Hugging Face `accelerate` (`device_map="auto"`).
   
   ```bash
   # Run full multi-task evaluation
   python evaluate.py --model_dir ./p2n-model --dataset_path data --num_samples 500
   
   # Run inference on a specific image for all tasks
   python inference.py --model_dir ./p2n-model --image plot.png --task all
   ```
