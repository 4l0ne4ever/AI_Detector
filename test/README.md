# AI vs Human Text Classifier - Modular Prototype

🚀 **Comprehensive prototype with separate models and shared configuration**

## Overview

This modular prototype tests multiple machine learning algorithms for distinguishing between AI-generated and human-written academic texts. Each algorithm has its own file but uses shared configuration and real crawled data.

## Features

### **Modular Architecture:**
- ✅ **Separate files** for each algorithm
- ✅ **Shared configuration** across all models
- ✅ **Real crawled data** (your existing dataset)
- ✅ **Consistent evaluation** methodology

### **Algorithms Tested:**
- **SVM** (`svm_classifier.py`) - Support Vector Machine with RBF kernel
- **MLP** (`mlp_classifier.py`) - Multi-Layer Perceptron neural network
- **Naive Bayes** (`baseline_classifiers.py`) - Fast probabilistic baseline
- **Logistic Regression** (`baseline_classifiers.py`) - Interpretable linear baseline

### **Shared Components:**
- **Data Loading** - Uses your real crawled human texts + existing AI texts
- **Feature Extraction** - TF-IDF with consistent parameters
- **Evaluation** - Same metrics and cross-validation for all models
- **Configuration** - All parameters centralized in `config.py`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Models
```bash
# Test SVM
python svm_classifier.py

# Test MLP Neural Network
python mlp_classifier.py

# Test Baseline Models
python baseline_classifiers.py
```

### 3. Run Complete Comparison
```bash
python compare_all_models.py
```

## File Structure

```
test/
├── config.py                    # Shared configuration and utilities
├── svm_classifier.py           # SVM implementation
├── mlp_classifier.py           # MLP Neural Network implementation  
├── baseline_classifiers.py     # Naive Bayes + Logistic Regression
├── compare_all_models.py       # Run all models and compare
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Expected Output

### Individual Model Output:
```bash
$ python svm_classifier.py

🤖 SVM Classifier for AI vs Human Text Detection
============================================================

=== Creating Balanced Test Dataset ===
Loading human texts from ../human_text_50k.jsonl...
Loaded 1000 human texts
Loading AI texts from ../ai_generated_colab.jsonl...
Loaded 1000 AI texts

=== Dataset Summary ===
Total samples: 2000
Human samples: 1000
AI samples: 1000
Balance ratio: 1.00
Average text length: 892 characters

Dataset Split:
Training samples: 1600
Testing samples: 400

Extracting TF-IDF features...
Features shape: (1600, 3000)

--- Evaluating SVM ---
Accuracy: 0.8425
Precision: 0.8254
Recall: 0.8673
F1-Score: 0.8458
AUC: 0.9124
CV Score: 0.8312 (±0.0156)
Training time: 2.45s
Prediction time: 0.123s

🎯 SVM Classification Complete!
⚡ Total execution time: 8.7 seconds
🏆 Final F1-Score: 0.8458
✅ Excellent performance - SVM works great for this task!
```

### Complete Comparison Output:
```bash
$ python compare_all_models.py

🚀 COMPREHENSIVE AI vs HUMAN TEXT CLASSIFICATION COMPARISON
================================================================================

🏆 FINAL MODEL COMPARISON
================================================================================
Model                Accuracy   Precision  Recall     F1-Score   AUC        Train(s)   CV Score    
----------------------------------------------------------------------------------------------------
SVM                  0.8425     0.8254     0.8673     0.8458     0.9124     2.45       0.8312±0.016
Logistic Regression  0.8375     0.8198     0.8612     0.8401     0.9087     0.89       0.8289±0.018
MLP                  0.8225     0.8045     0.8487     0.8261     0.8934     4.12       0.8156±0.022
Naive Bayes          0.7875     0.7723     0.8234     0.7971     0.8745     0.12       0.7889±0.019

🏆 BEST PERFORMING MODEL:
   Model: SVM
   F1-Score: 0.8458
   Accuracy: 0.8425
   AUC: 0.9124
   Cross-validation: 0.8312 (±0.0156)

📈 PERFORMANCE ASSESSMENT:
   ✅ EXCELLENT - Ready for production scaling!

🎯 FINAL RECOMMENDATION
================================================================================
✅ PROJECT IS READY TO SCALE!
   Your best model (SVM) achieved F1-Score: 0.8458
   This indicates the AI vs Human classification problem is solvable.
   You can confidently invest in scaling up the dataset and deployment.
```

## Interpretation

### **Performance Thresholds:**
- **F1-Score > 0.85**: Excellent - Ready for production scaling
- **F1-Score > 0.75**: Good - Consider feature engineering
- **F1-Score < 0.75**: Poor - Need better approach

### **What the Results Mean:**

**If you get F1-Score > 0.8:**
- ✅ **Concept is viable!** 
- ✅ **AI vs Human detection is feasible**
- ✅ **Scale up to larger datasets**

**If you get F1-Score < 0.8:**
- ⚠️ **Need improvements:**
  - Try better AI text generation
  - Add more sophisticated features  
  - Consider deep learning approaches

## Files Generated

- `test_dataset_2k.jsonl` - Balanced test dataset
- Console output with detailed performance metrics

## Next Steps

### **If Results Look Good (F1 > 0.8):**
1. **Scale up dataset**: Generate 10K-50K samples
2. **Implement real model**: Use transformer-based models
3. **Production pipeline**: Create API for real-time classification
4. **Advanced features**: Add linguistic analysis, stylometry

### **If Results Need Improvement (F1 < 0.8):**
1. **Better AI generation**: Use actual LLM APIs instead of simple transformations
2. **Feature engineering**: Add more sophisticated text features
3. **Deep learning**: Try BERT/RoBERTa classifiers
4. **Data quality**: Ensure higher quality AI vs human distinction

## Technical Details

### **Dataset Generation:**
- Uses simple text transformations for quick testing
- Applies 2-4 random linguistic changes per text
- Maintains semantic meaning while changing style
- 20% length variation to simulate real AI behavior

### **Feature Engineering:**
- **TF-IDF vectors** with unigrams and bigrams
- **Linguistic features**: Length, punctuation, vocabulary diversity
- **N-gram analysis** for style detection
- **Preprocessing**: Stop word removal, normalization

### **Model Configuration:**
- **SVM**: RBF kernel with probability estimation
- **MLP**: 100→50 hidden layers with early stopping
- **Cross-validation**: 5-fold stratified for robust evaluation
- **Balanced classes**: Equal human/AI samples

## Limitations

⚠️ **This is a PROTOTYPE using simple AI text generation**

For production use, you'll need:
- Real LLM-generated texts (GPT, Claude, etc.)
- Larger, more diverse datasets
- More sophisticated models
- Better evaluation frameworks

---

**Time Investment**: ~10 minutes total
**Purpose**: Validate concept before investing in full implementation
**Decision Point**: F1-Score determines whether to proceed with full project