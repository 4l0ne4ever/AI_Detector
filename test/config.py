#!/usr/bin/env python3
"""
Shared Configuration for AI Text Classification Models
All models use the same parameters and data preprocessing
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ========================
# SHARED PARAMETERS
# ========================

# Data Configuration
DATA_CONFIG = {
    'human_file': '../human_text_50k.jsonl',
    'ai_file': None,  # We'll look for existing AI generated files
    'test_size': 0.2,
    'random_state': 42,
    'samples_per_class': 1000,  # Small test size
}

# Feature Extraction Parameters
FEATURE_CONFIG = {
    'tfidf_max_features': 3000,
    'tfidf_ngram_range': (1, 2),  # Unigrams and bigrams
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.95,
    'stop_words': 'english',
    'use_scaler': True,
    'scaler_sparse_safe': True,
}

# Training Parameters
TRAINING_CONFIG = {
    'random_state': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'stratify': True,
    'shuffle': True,
}

# Evaluation Parameters
EVAL_CONFIG = {
    'pos_label': 'human',
    'average': 'binary',
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
    'verbose': True,
}

# Model-specific Parameters (shared base, can be overridden)
MODEL_PARAMS = {
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'random_state': 42,
    },
    'mlp': {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.01,
        'batch_size': 'auto',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
    },
    'naive_bayes': {
        'alpha': 1.0,
    },
    'logistic_regression': {
        'C': 1.0,
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs',
    }
}

# ========================
# SHARED DATA FUNCTIONS
# ========================

class DataLoader:
    """Shared data loading and preprocessing"""
    
    @staticmethod
    def load_human_texts(file_path: str, max_samples: int) -> List[Dict]:
        """Load human texts from crawled data"""
        print(f"Loading human texts from {file_path}...")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Human data file not found: {file_path}")
        
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # Shuffle for random sampling
        np.random.seed(42)
        np.random.shuffle(all_lines)
        
        for line in all_lines[:max_samples * 2]:  # Load extra for filtering
            try:
                entry = json.loads(line.strip())
                text = entry.get('text', '')
                if text and len(text) > 100:  # Filter very short texts
                    texts.append({
                        'text': text,
                        'label': 'human',
                        'length': len(text),
                        'source': 'human_crawled'
                    })
            except json.JSONDecodeError:
                continue
                
            if len(texts) >= max_samples:
                break
        
        print(f"Loaded {len(texts)} human texts")
        return texts[:max_samples]
    
    @staticmethod
    def load_ai_texts(max_samples: int) -> List[Dict]:
        """Load AI texts from existing generated files"""
        # Look for existing AI-generated files
        ai_files = [
            'ai_generated_colab.jsonl',
            '../ai_generated_colab.jsonl',
            'ai_generated.jsonl',
            '../ai_generated.jsonl'
        ]
        
        texts = []
        for file_path in ai_files:
            if Path(file_path).exists():
                print(f"Loading AI texts from {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            text = entry.get('text', '')
                            if text and len(text) > 100:
                                texts.append({
                                    'text': text,
                                    'label': 'ai',
                                    'length': len(text),
                                    'source': 'ai_generated'
                                })
                        except json.JSONDecodeError:
                            continue
                            
                        if len(texts) >= max_samples:
                            break
                break
        
        if not texts:
            print("Warning: No AI-generated files found. Using text transformations...")
            # Fallback: create simple AI texts for testing
            human_texts = DataLoader.load_human_texts(DATA_CONFIG['human_file'], max_samples)
            texts = DataLoader.create_simple_ai_texts(human_texts[:max_samples])
        
        print(f"Loaded {len(texts)} AI texts")
        return texts[:max_samples]
    
    @staticmethod
    def create_simple_ai_texts(human_texts: List[Dict]) -> List[Dict]:
        """Create simple AI texts using transformations (fallback)"""
        print("Creating simple AI texts using transformations...")
        
        ai_texts = []
        patterns = [
            lambda text: text.replace("we", "the researchers").replace("our", "their"),
            lambda text: text.replace("this paper", "this study").replace("shows", "demonstrates"),
            lambda text: text.replace("method", "approach").replace("results", "findings"),
            lambda text: text.replace("analysis", "examination").replace("technique", "methodology"),
        ]
        
        for i, human_entry in enumerate(human_texts):
            ai_text = human_entry['text']
            # Apply 2-3 random transformations
            selected_patterns = np.random.choice(patterns, size=np.random.randint(2, 4), replace=False)
            for pattern in selected_patterns:
                ai_text = pattern(ai_text)
            
            ai_texts.append({
                'text': ai_text,
                'label': 'ai',
                'length': len(ai_text),
                'source': 'simple_transformation'
            })
        
        return ai_texts
    
    @staticmethod
    def create_balanced_dataset(samples_per_class: int = 1000) -> Tuple[List[str], List[str]]:
        """Create balanced dataset with equal human and AI samples"""
        print(f"\n=== Creating Balanced Test Dataset ===")
        print(f"Target: {samples_per_class} samples per class ({samples_per_class * 2} total)")
        
        # Load data
        human_texts = DataLoader.load_human_texts(DATA_CONFIG['human_file'], samples_per_class)
        ai_texts = DataLoader.load_ai_texts(samples_per_class)
        
        # Ensure balance
        min_samples = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_samples]
        ai_texts = ai_texts[:min_samples]
        
        # Combine
        all_texts = human_texts + ai_texts
        np.random.seed(42)
        np.random.shuffle(all_texts)
        
        # Extract texts and labels
        texts = [entry['text'] for entry in all_texts]
        labels = [entry['label'] for entry in all_texts]
        
        # Stats
        human_count = labels.count('human')
        ai_count = labels.count('ai')
        avg_length = np.mean([len(text) for text in texts])
        
        print(f"\n=== Dataset Summary ===")
        print(f"Total samples: {len(texts)}")
        print(f"Human samples: {human_count}")
        print(f"AI samples: {ai_count}")
        print(f"Balance ratio: {human_count / ai_count:.2f}")
        print(f"Average text length: {avg_length:.0f} characters")
        
        return texts, labels

# ========================
# SHARED FEATURE EXTRACTION
# ========================

class FeatureExtractor:
    """Shared feature extraction for all models"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = None
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform training texts"""
        print("Extracting TF-IDF features...")
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=FEATURE_CONFIG['tfidf_max_features'],
            ngram_range=FEATURE_CONFIG['tfidf_ngram_range'],
            min_df=FEATURE_CONFIG['tfidf_min_df'],
            max_df=FEATURE_CONFIG['tfidf_max_df'],
            stop_words=FEATURE_CONFIG['stop_words']
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Scaling (sparse-safe)
        if FEATURE_CONFIG['use_scaler']:
            self.scaler = StandardScaler(with_mean=False)  # Sparse-safe
            features = self.scaler.fit_transform(tfidf_matrix)
        else:
            features = tfidf_matrix
        
        self.is_fitted = True
        print(f"Features shape: {features.shape}")
        return features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using fitted extractors"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        if self.scaler:
            features = self.scaler.transform(tfidf_matrix)
        else:
            features = tfidf_matrix
            
        return features

# ========================
# SHARED EVALUATION
# ========================

class ModelEvaluator:
    """Shared evaluation functions for all models"""
    
    @staticmethod
    def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
        """Comprehensive model evaluation"""
        import time
        
        print(f"\n--- Evaluating {model_name} ---")
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Probabilities for AUC (if available)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]  # Probability of 'human' class
        except:
            y_proba = None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=EVAL_CONFIG['pos_label'])
        recall = recall_score(y_test, y_pred, pos_label=EVAL_CONFIG['pos_label'])
        f1 = f1_score(y_test, y_pred, pos_label=EVAL_CONFIG['pos_label'])
        
        # AUC
        auc = None
        if y_proba is not None:
            y_test_binary = [1 if label == 'human' else 0 for label in y_test]
            try:
                auc = roc_auc_score(y_test_binary, y_proba)
            except:
                pass
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=TRAINING_CONFIG['cv_folds'], 
                                  scoring='accuracy')
        
        # Results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time,
            'predict_time': predict_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Print results
        if EVAL_CONFIG['verbose']:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if auc:
                print(f"AUC: {auc:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"Training time: {train_time:.2f}s")
            print(f"Prediction time: {predict_time:.3f}s")
        
        return results
    
    @staticmethod
    def save_results(results: Dict, output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dumps(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")

# Global instances (shared across all models)
data_loader = DataLoader()
feature_extractor = FeatureExtractor()
evaluator = ModelEvaluator()

print("✅ Shared configuration loaded successfully!")
print(f"Models will use: {list(MODEL_PARAMS.keys())}")
print(f"TF-IDF features: {FEATURE_CONFIG['tfidf_max_features']} max features")
print(f"Test samples per class: {DATA_CONFIG['samples_per_class']}")