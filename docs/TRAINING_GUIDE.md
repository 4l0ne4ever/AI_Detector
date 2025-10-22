# Human vs AI Text Classification - Complete Training Pipeline

## After AI Text Generation: Step-by-Step Guide

Once you've generated your 50k AI texts using the Google Colab notebook, follow these steps to train and evaluate your classification models.

---

## Step 1: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt

# Optional: For BERT training (requires more memory/time)
pip install torch transformers datasets accelerate
```

---

## Step 2: Data Preprocessing

Combine your human and AI datasets, clean the data, and create train/validation/test splits.

```bash
# Make sure you have both files:
# - data/crawl/human_text_50k.jsonl (your human texts)
# - data/generate/ai_text_50k_colab.jsonl (generated AI texts)

python3 preprocessing/preprocessing.py
```

**Expected Output:**
- `processed_data/` directory with train/val/test splits
- Data analysis visualizations
- Preprocessing statistics

**Time:** ~2-5 minutes for 100k texts

---

## Step 3: Train Multiple Models

Train traditional ML models (SVM, Random Forest, Naive Bayes, Logistic Regression) and optionally BERT.

```bash
python3 training/train_models.py
```

**Expected Output:**
- `trained_models/` directory with saved model files
- `model_results/` directory with training metrics
- Performance comparison of all models

**Time:** 
- Traditional models: 5-15 minutes
- BERT (if enabled): 2-4 hours

---

## Step 4: Detailed Evaluation

Generate comprehensive evaluation reports with visualizations.

```bash
python3 evaluation/evaluate_models.py
```

**Expected Output:**
- ROC curves and Precision-Recall curves
- Confusion matrices
- Feature importance analysis
- Error analysis with misclassified examples
- Detailed evaluation report

**Time:** ~1-3 minutes

---

## Step 5: Use Your Classifier

Now you can classify any new text as human or AI-written!

### Interactive Mode (Recommended for Testing)
```bash
python3 prediction/predict_text.py
```

### Command Line Usage
```bash
# Classify a specific text
python3 prediction/predict_text.py --text "Your scientific abstract here..."

# Classify text from a file
python3 prediction/predict_text.py --file "path/to/text.txt"

# Batch process multiple texts
python3 prediction/predict_text.py --file "texts.txt" --batch --output "results.json"
```

---

## Final Project Structure

After completing all steps:

```
IT3190/
├── requirements.txt                  # Dependencies
├── data/
│   ├── crawl/                        # Crawled human data
│   │   ├── crawl.py                  # Original data crawler
│   │   ├── human_text_50k.jsonl      # Human abstracts (50k)
│   │   └── human_text_100k.jsonl     # Human abstracts (100k backup)
│   └── generate/                     # Generated AI data
│       ├── local_backup_generation.py # Local AI generation script
│       └── ai_text_50k_colab.jsonl   # Generated AI texts
│
├── preprocessing/
│   └── preprocessing.py              # Data preprocessing pipeline
│
├── training/
│   └── train_models.py               # Multi-model training
│
├── evaluation/
│   └── evaluate_models.py            # Detailed evaluation
│
├── prediction/
│   └── predict_text.py               # Final classifier interface
│
├── docs/
│   ├── AI_Text_Generation_Colab.ipynb # Colab notebook
│   ├── TRAINING_GUIDE.md             # This guide
│   ├── PROJECT_STATUS.md             # Project overview
│   └── COLAB_INSTRUCTIONS.md         # AI generation guide
│
├── processed_data/               # Created after preprocessing
│   ├── train_data.jsonl
│   ├── val_data.jsonl
│   ├── test_data.jsonl
│   └── data_analysis.png
│
├── trained_models/               # Created after training
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   └── ...
│
└── model_results/                # Created after training
    ├── training_results.json
    ├── results_summary.csv
    └── detailed_evaluation/
```

---

## Expected Performance

Based on similar human vs AI classification tasks:

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| **SVM** | 85-92% | 2-5 minutes |
| **Random Forest** | 83-89% | 1-3 minutes |
| **Logistic Regression** | 84-90% | 1-2 minutes |
| **Naive Bayes** | 80-86% | <1 minute |
| **BERT** | 90-95% | 2-4 hours |

---

## Troubleshooting

### Common Issues:

1. **Missing AI texts file**
   ```
   ERROR: AI file not found: data/ai_text_50k_colab.jsonl
   ```
   **Solution:** Complete the Google Colab AI generation first

2. **Memory errors during training**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution:** Reduce dataset size or use smaller models

3. **BERT training fails**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** Traditional models work well, BERT is optional

4. **Import errors**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Solution:** Install requirements: `pip install -r requirements.txt`

---

## Tips for Best Results

1. **Data Quality**: Ensure both human and AI texts are high-quality scientific abstracts
2. **Balanced Dataset**: Keep roughly equal numbers of human and AI texts
3. **Text Length**: Filter out very short (<50 chars) or very long (>5000 chars) texts
4. **Model Selection**: Start with SVM - it usually performs best on text classification
5. **Feature Engineering**: The TF-IDF features work well for this task

---

## Next Steps & Extensions

After your basic classifier is working:

1. **Feature Enhancement**: Try different text features (word embeddings, linguistic features)
2. **Model Ensembling**: Combine multiple models for better performance  
3. **Domain Adaptation**: Train on specific scientific domains
4. **Real-world Testing**: Test on recent AI-generated papers
5. **Web Interface**: Build a simple web app for easy access

---

## Performance Monitoring

Keep track of your model's performance:

- **Accuracy**: Overall correct predictions
- **Precision**: How many AI predictions were actually AI
- **Recall**: How many actual AI texts were caught
- **F1-Score**: Balanced measure of precision and recall
- **AUC**: Area under ROC curve (discrimination ability)

The classifier works best on scientific abstracts but can be adapted for other text types with retraining.

---

**Congratulations! You now have a complete human vs AI text classification system!**
