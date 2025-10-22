#!/usr/bin/env python3
"""
AI vs Human Text Classifier Prototype
Tests SVM, MLP, and baseline models for proof of concept
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# For embeddings (if available)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

class TextFeatureExtractor:
    """Extract features from text for ML models"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.embedding_model = None
        
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features like length, punctuation, etc."""
        features = []
        
        for text in texts:
            text_features = [
                len(text),  # Character length
                len(text.split()),  # Word count
                len([s for s in text.split('.') if s.strip()]),  # Sentence count
                text.count(',') / len(text) * 1000,  # Comma density
                text.count('!') + text.count('?'),  # Exclamation/question marks
                sum(1 for c in text if c.isupper()) / len(text) * 100,  # Uppercase ratio
                text.count('the') + text.count('and') + text.count('of'),  # Common words
                len(set(text.lower().split())) / len(text.split()) if text.split() else 0,  # Vocab diversity
            ]
            features.append(text_features)
            
        return np.array(features)
    
    def extract_tfidf_features(self, texts: List[str], max_features: int = 5000) -> Tuple[np.ndarray, Any]:
        """Extract TF-IDF features"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,
                max_df=0.95
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
        return tfidf_matrix.toarray(), self.tfidf_vectorizer
    
    def extract_embedding_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentence embeddings (if available)"""
        if not EMBEDDINGS_AVAILABLE:
            print("Warning: sentence-transformers not available, using random embeddings for demo")
            return np.random.rand(len(texts), 384)  # Simulate embedding dimensions
        
        if self.embedding_model is None:
            print("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        embeddings = self.embedding_model.encode(texts)
        return embeddings

class AITextClassifier:
    """Main classifier class with multiple algorithms"""
    
    def __init__(self):
        self.feature_extractor = TextFeatureExtractor()
        self.models = {}
        self.results = {}
        
    def load_data(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Load test dataset"""
        print(f"Loading dataset from {file_path}...")
        
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    texts.append(entry['text'])
                    labels.append(entry['label'])
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    
        print(f"Loaded {len(texts)} texts")
        print(f"Human: {labels.count('human')}, AI: {labels.count('ai')}")
        
        return texts, labels
    
    def create_svm_model(self) -> Pipeline:
        """Create SVM model with TF-IDF features"""
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))),
            ('scaler', StandardScaler(with_mean=False)),  # Sparse-compatible scaling
            ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
        ])
    
    def create_mlp_model(self) -> Pipeline:
        """Create MLP model with combined features"""
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
            ('scaler', StandardScaler(with_mean=False)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
    
    def create_baseline_models(self) -> Dict[str, Pipeline]:
        """Create baseline models for comparison"""
        return {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')),
                ('nb', MultinomialNB(alpha=1.0))
            ]),
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
                ('scaler', StandardScaler(with_mean=False)),
                ('lr', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
            ])
        }
    
    def evaluate_model(self, model: Any, X_train: Any, X_test: Any, 
                      y_train: List[str], y_test: List[str], model_name: str) -> Dict:
        """Evaluate a single model"""
        print(f"\n--- Evaluating {model_name} ---")
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Probabilities for AUC
        try:
            y_proba = model.predict_proba(X_test)[:, 1]  # Probability of 'human' class
        except:
            y_proba = None
            
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='human')
        recall = recall_score(y_test, y_pred, pos_label='human')
        f1 = f1_score(y_test, y_pred, pos_label='human')
        
        # AUC if probabilities available
        auc = None
        if y_proba is not None:
            y_test_binary = [1 if label == 'human' else 0 for label in y_test]
            try:
                auc = roc_auc_score(y_test_binary, y_proba)
            except:
                auc = None
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
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
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
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
    
    def run_comparison(self, file_path: str) -> Dict:
        """Run complete model comparison"""
        print("🤖 AI vs Human Text Classification Prototype")
        print("=" * 60)
        
        # Load data
        texts, labels = self.load_data(file_path)
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nDataset split:")
        print(f"Training: {len(X_train_text)} samples")
        print(f"Testing: {len(X_test_text)} samples")
        
        # Create models
        models = {
            'SVM': self.create_svm_model(),
            'MLP': self.create_mlp_model(),
            **self.create_baseline_models()
        }
        
        # Evaluate each model
        all_results = []
        
        for name, model in models.items():
            try:
                results = self.evaluate_model(
                    model, X_train_text, X_test_text, y_train, y_test, name
                )
                all_results.append(results)
                self.models[name] = model
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        # Summary comparison
        self.print_comparison_summary(all_results)
        
        return {'models': self.models, 'results': all_results}
    
    def print_comparison_summary(self, results: List[Dict]):
        """Print comparison summary"""
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        # Create comparison table
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('f1_score', ascending=False)
        
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 80)
        
        for _, row in df_results.iterrows():
            auc_str = f"{row['auc']:.4f}" if row['auc'] else "N/A"
            print(f"{row['model_name']:<20} {row['accuracy']:<10.4f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1_score']:<10.4f} {auc_str:<10}")
        
        # Best model
        best_model = df_results.iloc[0]
        print(f"\n🏆 Best Model: {best_model['model_name']}")
        print(f"   F1-Score: {best_model['f1_score']:.4f}")
        print(f"   Cross-validation: {best_model['cv_mean']:.4f} (±{best_model['cv_std']:.4f})")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        if best_model['f1_score'] > 0.85:
            print("   ✅ Excellent performance - Ready to scale up!")
        elif best_model['f1_score'] > 0.75:
            print("   ⚠️  Good performance - Consider feature engineering")
        else:
            print("   ❌ Poor performance - Need better approach")
        
        print(f"\n⏱️  Training Time Analysis:")
        fastest = min(results, key=lambda x: x['train_time'])
        print(f"   Fastest: {fastest['model_name']} ({fastest['train_time']:.2f}s)")
        
        return df_results

def main():
    """Main execution"""
    classifier = AITextClassifier()
    
    # Dataset file
    dataset_file = "test_dataset_2k.jsonl"
    
    if not Path(dataset_file).exists():
        print(f"❌ Dataset not found: {dataset_file}")
        print("Please run: python create_test_dataset.py first")
        return
    
    # Run comparison
    start_time = time.time()
    results = classifier.run_comparison(dataset_file)
    total_time = time.time() - start_time
    
    print(f"\n⚡ Total execution time: {total_time:.1f} seconds")
    
    # Recommendations
    print(f"\n🎯 Next Steps:")
    print(f"1. If F1-score > 0.8: Scale up with larger dataset")
    print(f"2. If F1-score < 0.8: Improve features or try deep learning")
    print(f"3. Consider ensemble methods for production")
    print(f"4. Implement real-time classification API")
    
    return results

if __name__ == "__main__":
    main()
