#!/usr/bin/env python3
"""
Machine Learning Training Pipeline
=================================

This script trains multiple models for human vs AI text classification:
- Traditional ML: SVM, Random Forest, Naive Bayes
- Deep Learning: BERT fine-tuning

Usage:
    python3 train_models.py
"""

import json
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report, roc_auc_score)
from sklearn.pipeline import Pipeline

# Deep Learning (optional)
try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                            Trainer, TrainingArguments, EarlyStoppingCallback)
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install torch transformers datasets")

class ModelTrainer:
    def __init__(self, data_dir="../processed_data"):
        self.data_dir = data_dir
        self.results_dir = "../model_results"
        self.models_dir = "../trained_models"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Results storage
        self.results = {}
        
    def load_processed_data(self):
        """Load preprocessed data"""
        
        print("Loading processed data...")
        
        try:
            # Load training data
            self.train_data = pd.read_json(f"{self.data_dir}/train_data.jsonl", lines=True)
            self.val_data = pd.read_json(f"{self.data_dir}/val_data.jsonl", lines=True)
            self.test_data = pd.read_json(f"{self.data_dir}/test_data.jsonl", lines=True)
            
            print(f"Loaded data:")
            print(f"   Train: {len(self.train_data):,} texts")
            print(f"   Val: {len(self.val_data):,} texts")
            print(f"   Test: {len(self.test_data):,} texts")
            
            # Convert labels to binary (0=human, 1=ai)
            label_map = {'human': 0, 'ai': 1}
            for data in [self.train_data, self.val_data, self.test_data]:
                data['label_binary'] = data['label'].map(label_map)
            
            return True
            
        except FileNotFoundError as e:
            print(f"Could not load processed data: {e}")
            print("Run preprocessing.py first!")
            return False
    
    def train_traditional_models(self):
        """Train traditional ML models"""
        
        print("\\nTraining Traditional ML Models...")
        print("="*40)
        
        # Prepare data
        X_train = self.train_data['text']
        y_train = self.train_data['label_binary']
        X_val = self.val_data['text']
        y_val = self.val_data['label_binary']
        
        # Models to train
        models = {
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]),
            
            'Random_Forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ]),
            
            'Naive_Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
                ('nb', MultinomialNB(alpha=1.0))
            ]),
            
            'Logistic_Regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ])
        }
        
        for name, model in models.items():
            print(f"\\nTraining {name}...")
            
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on validation set
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of AI class
            
            # Calculate metrics
            train_time = time.time() - start_time
            accuracy = accuracy_score(y_val, val_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred, average='binary')
            auc = roc_auc_score(y_val, val_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'train_time': train_time,
                'type': 'traditional'
            }
            
            print(f"   {name} completed:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      F1: {f1:.4f}")
            print(f"      AUC: {auc:.4f}")
            print(f"      Train time: {train_time:.1f}s")
            
            # Save model
            model_file = f"{self.models_dir}/{name.lower()}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
        print("\\nTraditional models training completed!")
        
    def train_bert_model(self):
        """Train BERT model for comparison"""
        
        if not TRANSFORMERS_AVAILABLE:
            print("\\nSkipping BERT training - transformers not available")
            return
        
        print("\\nTraining BERT Model...")
        print("="*30)
        
        try:
            # Model setup
            model_name = "distilbert-base-uncased"  # Smaller, faster BERT
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            
            # Tokenize data
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'], 
                    truncation=True, 
                    padding='max_length',
                    max_length=512
                )
            
            # Create datasets
            train_dataset = Dataset.from_pandas(
                self.train_data[['text', 'label_binary']].rename(columns={'label_binary': 'labels'})
            )
            val_dataset = Dataset.from_pandas(
                self.val_data[['text', 'label_binary']].rename(columns={'label_binary': 'labels'})
            )
            
            # Tokenize
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{self.models_dir}/bert_checkpoints",
                num_train_epochs=3,
                per_device_train_batch_size=8,  # Small batch for memory efficiency
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{self.results_dir}/bert_logs",
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=500,
                save_strategy="steps",
                save_steps=500,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            print("Starting BERT training...")
            start_time = time.time()
            
            # Train
            trainer.train()
            
            train_time = time.time() - start_time
            
            # Evaluate
            eval_results = trainer.evaluate(val_dataset)
            
            # Predictions for metrics
            predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            # Store results
            self.results['BERT'] = {
                'model': trainer.model,
                'tokenizer': tokenizer,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': 'N/A',  # Would need probability predictions
                'train_time': train_time,
                'eval_loss': eval_results['eval_loss'],
                'type': 'deep_learning'
            }
            
            print(f"   BERT completed:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      F1: {f1:.4f}")
            print(f"      Train time: {train_time/60:.1f} min")
            
            # Save model
            trainer.save_model(f"{self.models_dir}/bert_model")
            
        except Exception as e:
            print(f"BERT training failed: {e}")
            print("This might be due to memory constraints or missing dependencies")
    
    def evaluate_on_test_set(self):
        """Evaluate all models on test set"""
        
        print("\\nFinal Evaluation on Test Set...")
        print("="*35)
        
        X_test = self.test_data['text']
        y_test = self.test_data['label_binary']
        
        final_results = {}
        
        for name, result in self.results.items():
            if result['type'] == 'traditional':
                model = result['model']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate test metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary'
                )
                test_auc = roc_auc_score(y_test, y_pred_proba)
                
                final_results[name] = {
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'test_auc': test_auc,
                    'train_time': result['train_time']
                }
                
                print(f"\\n{name} Test Results:")
                print(f"   Accuracy: {test_accuracy:.4f}")
                print(f"   Precision: {test_precision:.4f}")
                print(f"   Recall: {test_recall:.4f}")
                print(f"   F1: {test_f1:.4f}")
                print(f"   AUC: {test_auc:.4f}")
        
        return final_results
    
    def save_results(self, final_results):
        """Save training results"""
        
        print("\\nSaving results...")
        
        # Combine validation and test results
        combined_results = {}
        for name in self.results.keys():
            if name in final_results:
                combined_results[name] = {
                    **{f"val_{k}": v for k, v in self.results[name].items() 
                       if k not in ['model', 'tokenizer']},
                    **final_results[name],
                    'model_type': self.results[name]['type']
                }
        
        # Save results JSON
        results_file = f"{self.results_dir}/training_results.json"
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Create results summary table
        summary_df = pd.DataFrame(combined_results).T
        summary_df = summary_df.round(4)
        
        # Save CSV
        summary_df.to_csv(f"{self.results_dir}/results_summary.csv")
        
        # Print summary table
        print("\\nFinal Results Summary:")
        print("="*50)
        print(summary_df[['test_accuracy', 'test_f1', 'test_auc', 'train_time']].to_string())
        
        # Best model
        best_model = summary_df.loc[summary_df['test_f1'].idxmax()]
        print(f"\\nBest Model: {best_model.name}")
        print(f"   Test F1: {best_model['test_f1']:.4f}")
        print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}")
        
        print(f"\\nResults saved to {self.results_dir}/")
        
        return best_model.name

def main():
    """Main training pipeline"""
    
    print("Human vs AI Text Classification - Model Training")
    print("="*52)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load processed data
    if not trainer.load_processed_data():
        return
    
    # Train traditional models
    trainer.train_traditional_models()
    
    # Train BERT model (if available)
    trainer.train_bert_model()
    
    # Final evaluation on test set
    final_results = trainer.evaluate_on_test_set()
    
    # Save results
    best_model = trainer.save_results(final_results)
    
    print("\\nTraining pipeline completed successfully!")
    print(f"\\nNext steps:")
    print(f"   1. Review results in model_results/ directory")
    print(f"   2. Use the best model ({best_model}) for predictions")
    print(f"   3. Run evaluation script for detailed analysis")

if __name__ == "__main__":
    main()