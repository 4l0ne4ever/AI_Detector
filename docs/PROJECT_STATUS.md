# Human vs AI Text Classification Project

## Current Status

### Data Collection - COMPLETED
- **Human texts**: 50,019 scientific paper abstracts from arXiv
- **File**: `data/crawl/human_text_50k.jsonl` (76MB)
- **Quality**: High-quality academic abstracts with metadata
- **Coverage**: Multiple research domains (cs, physics, math, etc.)

### AI Text Generation - READY TO START
- **Target**: 50,000 AI-generated texts to match human dataset
- **Method**: Google Colab with GPU acceleration (RECOMMENDED)
- **Alternative**: Local generation using Ollama (backup)

## Project Structure
```
IT3190/
├── requirements.txt                  # Dependencies
├── data/
│   ├── crawl/                        # Crawled human data
│   │   ├── crawl.py                  # Original data crawler
│   │   ├── human_text_50k.jsonl      # Your 50k human texts
│   │   └── human_text_100k.jsonl     # Backup with 21k additional texts
│   └── generate/                     # Generated AI data
│       ├── local_backup_generation.py # Local AI generation script
│       └── (ai_text_50k_colab.jsonl after generation)
├── preprocessing/
│   └── preprocessing.py              # Data preprocessing pipeline
├── training/
│   └── train_models.py               # Multi-model training
├── evaluation/
│   └── evaluate_models.py            # Detailed evaluation
├── prediction/
│   └── predict_text.py               # Final classifier interface
├── docs/
│   ├── AI_Text_Generation_Colab.ipynb # Main GPU-accelerated generation
│   ├── COLAB_INSTRUCTIONS.md         # Setup guide for Colab
│   ├── PROJECT_STATUS.md             # This status file
│   └── TRAINING_GUIDE.md             # Complete training guide
└── logs/                             # Crawling logs
    └── arxiv_crawl.log
```

## Next Steps

### Immediate Action: AI Text Generation
1. **Upload to Google Colab** (RECOMMENDED - fastest method)
   - Use `docs/AI_Text_Generation_Colab.ipynb`
   - Follow `docs/COLAB_INSTRUCTIONS.md`
   - Expected time: 17-28 hours for 50k texts
   - Cost: FREE

2. **Alternative: Local Generation** (backup method)
   - Use `python3 data/generate/local_backup_generation.py`
   - Generate ~1k texts locally while Colab runs
   - Much slower but can run in parallel

### After AI Generation Complete:
3. **Data Preprocessing**
   - Run `python3 preprocessing/preprocessing.py`
   - Combine human and AI datasets
   - Split into train/validation/test sets
   - Text cleaning and tokenization

4. **Model Training**
   - Run `python3 training/train_models.py`
   - Traditional ML: SVM, Random Forest, Naive Bayes
   - Deep Learning: BERT fine-tuning
   - Compare performance metrics

5. **Evaluation**
   - Run `python3 evaluation/evaluate_models.py`
   - Accuracy, Precision, Recall, F1-score
   - ROC curves and confusion matrices
   - Feature importance analysis

## Performance Expectations

| Method | Speed | Time for 50k | Cost |
|--------|-------|-------------|------|
| **Google Colab (T4 GPU)** | 30-50/min | 17-28 hours | FREE |
| Local Ollama | 4-6/min | ~7-8 days | FREE |
| Hybrid approach | Mixed | 20-30 hours | FREE |

## Recommendations

1. **Start with Google Colab immediately** - it's the fastest free option
2. **Keep your MacBook available** for other tasks while Colab runs
3. **Monitor progress** - Colab saves automatically every 200 generations
4. **Plan training phase** - gather preprocessing and training scripts

## Project Highlights

- **Quality Data**: Authentic scientific abstracts, not summaries
- **Scale**: 50k+ samples per class for robust training  
- **Diversity**: Multiple academic domains and writing styles
- **Free Methods**: Zero-cost approach using Colab GPU acceleration
- **Automated**: Scripts handle resumption and progress tracking

---

**Ready to proceed with AI text generation using Google Colab!**