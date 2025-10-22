# Human vs AI Text Classification Project

A complete machine learning pipeline to classify scientific paper abstracts as human-written or AI-generated.

## Project Overview

This project provides:
- Data collection of 50k+ human-written scientific abstracts from arXiv
- AI text generation using Google Colab with GPU acceleration
- Comprehensive training pipeline with multiple ML algorithms
- Detailed evaluation and visualization tools
- Easy-to-use prediction interface

## Quick Start

### 1. Data Collection (COMPLETED)
- 50,019 human scientific abstracts already collected
- Located in `data/crawl/human_text_50k.jsonl`

### 2. AI Text Generation
```bash
# Upload docs/AI_Text_Generation_Colab.ipynb to Google Colab
# Follow instructions in docs/COLAB_INSTRUCTIONS.md
# Expected time: 17-28 hours for 50k AI texts
```

### 3. Training Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python3 preprocessing/preprocessing.py

# Train models
python3 training/train_models.py

# Evaluate performance
python3 evaluation/evaluate_models.py

# Use classifier
python3 prediction/predict_text.py
```

## Project Structure

```
IT3190/
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── data/                             # Training data and collection scripts
│   ├── crawl/                        # Crawled human data
│   │   ├── crawl.py                  # Original data crawler
│   │   ├── human_text_50k.jsonl      # Human abstracts (50k)
│   │   └── human_text_100k.jsonl     # Human abstracts (100k backup)
│   └── generate/                     # Generated AI data
│       ├── local_backup_generation.py # Local AI generation script
│       └── ai_text_50k_colab.jsonl   # Generated AI texts (after Colab)
├── preprocessing/                    # Data preprocessing
│   └── preprocessing.py
├── training/                         # Model training
│   └── train_models.py
├── evaluation/                       # Model evaluation
│   └── evaluate_models.py
├── prediction/                       # Text classification
│   └── predict_text.py
├── docs/                             # Documentation
│   ├── AI_Text_Generation_Colab.ipynb
│   ├── TRAINING_GUIDE.md             # Complete step-by-step guide
│   ├── PROJECT_STATUS.md             # Current status
│   └── COLAB_INSTRUCTIONS.md         # Colab setup guide
├── processed_data/                   # Created after preprocessing
├── trained_models/                   # Created after training
└── model_results/                    # Created after evaluation
```

## Documentation

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Complete step-by-step training instructions
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current project status and next steps
- **[COLAB_INSTRUCTIONS.md](docs/COLAB_INSTRUCTIONS.md)** - Google Colab setup guide

## Expected Performance

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| SVM | 85-92% | 2-5 minutes |
| Random Forest | 83-89% | 1-3 minutes |
| Logistic Regression | 84-90% | 1-2 minutes |
| Naive Bayes | 80-86% | <1 minute |
| BERT | 90-95% | 2-4 hours |

## Usage Examples

### Interactive Classification
```bash
python3 prediction/predict_text.py
# Enter text when prompted
```

### Command Line Classification
```bash
# Classify specific text
python3 prediction/predict_text.py --text "Your scientific abstract here"

# Classify from file
python3 prediction/predict_text.py --file "paper_abstract.txt"

# Batch processing
python3 prediction/predict_text.py --file "abstracts.txt" --batch --output "results.json"
```

## Requirements

- Python 3.7+
- See `requirements.txt` for complete dependencies
- Optional: GPU for BERT training (significant speedup)

## Installation

```bash
# Clone or download the project
git clone <repository-url>  # or download ZIP

# Install dependencies
pip install -r requirements.txt

# Optional: For BERT training
pip install torch transformers datasets accelerate
```

## Next Steps

1. **Generate AI texts**: Use Google Colab notebook for fast GPU generation
2. **Train models**: Run the complete training pipeline (20-25 minutes)
3. **Evaluate performance**: Analyze model performance with detailed metrics
4. **Start classifying**: Use your trained model to classify any scientific text

## Support

For detailed instructions, see `docs/TRAINING_GUIDE.md`

## License

This project is for educational purposes as part of IT3190 coursework.