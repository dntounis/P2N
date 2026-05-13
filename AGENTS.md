# Agent Steering Instructions

If you are an AI agent working in this repository, please adhere to the following rules:

1. **Architecture Focus:** The model architecture centers around `naver-clova-ix/donut-base`. Do not switch to different base vision models without explicit permission.
2. **Environment:** Training is optimized for CUDA clusters. Avoid re-introducing Apple Silicon (MPS) fallbacks unless specifically testing locally.
3. **Data Pipeline:** Ground truth data is generated in Hugging Face's `imagefolder` `metadata.jsonl` format. The parsing task prefix is `<s_p2n>`. Do not change this prefix.
4. **Data Generation:** `generate_data.py` uses Matplotlib and groups outputs into type-specific subfolders (`data/images/[plot_type]/image_XXXXX.png`). Ensure this logic remains intact when adding new plot generators.
5. **Evaluation:** When expanding `evaluate.py`, focus on extracting and parsing the JSON output string and calculating mathematical distance (MAE/RMSE) between the predicted data points and ground truth data points, not just simple text similarity metrics.
