# Google Colab AI Text Generation

## Quick Start Guide

### 1. Setup Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the `AI_Text_Generation_Colab.ipynb` file
3. Change Runtime → Hardware Accelerator → **GPU** (this is crucial!)

### 2. Upload Your Data
- Upload your `data/crawl/human_text_50k.jsonl` file when prompted
- The notebook will verify the upload and show statistics

### 3. Run the Generation
- Execute all cells in order (Runtime → Run All)
- The process will take approximately **17-28 hours** for 50k texts
- Speed: ~30-50 texts/minute with GPU acceleration

### 4. Download Results
- The generated file `ai_text_50k_colab.jsonl` will be automatically downloaded
- Place it in your local `data/generate/` directory

## Performance Expectations

| GPU Type | Speed | Time for 50k |
|----------|-------|-------------|
| T4 (Free) | ~30 texts/min | ~28 hours |
| V100 | ~45 texts/min | ~18 hours |
| A100 | ~50+ texts/min | ~17 hours |

## Troubleshooting

### Common Issues:
1. **"CUDA out of memory"**: The notebook automatically adjusts batch size based on GPU memory
2. **Disconnection**: The notebook saves progress automatically - just resume by re-running the generation cell
3. **Slow speed**: Make sure GPU is enabled in Runtime settings

### Tips:
- Keep the Colab tab active to prevent disconnection
- The notebook can resume from interruptions automatically
- Generated texts are saved incrementally every 200 samples

## File Structure After Generation
```
data/
├── crawl/
│   ├── human_text_50k.jsonl      # Your original human texts
│   └── human_text_100k.jsonl     # Backup human texts
└── generate/
    └── ai_text_50k_colab.jsonl   # Generated AI texts from Colab
```

## Next Steps
After getting your AI texts:
1. Follow the TRAINING_GUIDE.md for complete setup
2. Run preprocessing: `python3 preprocessing/preprocessing.py`
3. Train models: `python3 training/train_models.py`
4. Evaluate: `python3 evaluation/evaluate_models.py`
5. Use classifier: `python3 prediction/predict_text.py`

---

**Cost**: Completely free with Google Colab
**Quality**: High-quality academic style AI texts suitable for training classifiers