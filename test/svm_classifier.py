#!/usr/bin/env python3
"""
SVM Classifier for AI vs Human Text Detection
Uses shared configuration and data preprocessing
"""

import json
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from config import (
    DATA_CONFIG, TRAINING_CONFIG, MODEL_PARAMS, 
    data_loader, feature_extractor, evaluator
)

class SVMClassifier:
    """SVM-based AI text classifier"""
    
    def __init__(self):
        self.model = None
        self.results = None
        
    def create_model(self):
        """Create SVM model with shared parameters"""
        params = MODEL_PARAMS['svm']
        self.model = SVC(**params)
        print(f"SVM Model created with parameters: {params}")
        return self.model
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline"""
        print("🤖 SVM Classifier for AI vs Human Text Detection")
        print("=" * 60)
        
        # Load data using shared loader
        texts, labels = data_loader.create_balanced_dataset(
            DATA_CONFIG['samples_per_class']
        )
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=labels if TRAINING_CONFIG['stratify'] else None,
            shuffle=TRAINING_CONFIG['shuffle']
        )
        
        print(f"\nDataset Split:")
        print(f"Training samples: {len(X_train_text)}")
        print(f"Testing samples: {len(X_test_text)}")
        
        # Feature extraction using shared extractor
        print(f"\nExtracting features...")
        X_train = feature_extractor.fit_transform(X_train_text)
        X_test = feature_extractor.transform(X_test_text)
        
        # Create and evaluate model
        self.create_model()
        self.results = evaluator.evaluate_model(
            self.model, X_train, X_test, y_train, y_test, "SVM"
        )
        
        return self.results
    
    def save_results(self, filename: str = "svm_results.json"):
        """Save results to file"""
        if self.results:
            evaluator.save_results(self.results, filename)
        else:
            print("No results to save. Run train_and_evaluate() first.")
    
    def get_feature_importance(self, top_k: int = 20):
        """Get most important features (if available)"""
        if not self.model or not feature_extractor.is_fitted:
            print("Model must be trained first")
            return None
            
        try:
            # For SVM, we can't directly get feature importance
            # But we can analyze support vectors or coefficients (for linear kernel)
            if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                feature_names = feature_extractor.tfidf_vectorizer.get_feature_names_out()
                coef = self.model.coef_[0]
                
                # Get top positive and negative coefficients
                top_positive_idx = coef.argsort()[-top_k:][::-1]
                top_negative_idx = coef.argsort()[:top_k]
                
                print(f"\nTop {top_k} features for Human class:")
                for idx in top_positive_idx:
                    print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
                
                print(f"\nTop {top_k} features for AI class:")
                for idx in top_negative_idx:
                    print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
                
                return {
                    'human_features': [(feature_names[idx], coef[idx]) for idx in top_positive_idx],
                    'ai_features': [(feature_names[idx], coef[idx]) for idx in top_negative_idx]
                }
            else:
                print("Feature importance not available for RBF kernel SVM")
                return None
                
        except Exception as e:
            print(f"Error analyzing features: {e}")
            return None

def main():
    """Main execution for SVM classifier"""
    classifier = SVMClassifier()
    
    # Train and evaluate
    start_time = time.time()
    results = classifier.train_and_evaluate()
    total_time = time.time() - start_time
    
    print(f"\n🎯 SVM Classification Complete!")
    print(f"⚡ Total execution time: {total_time:.1f} seconds")
    print(f"🏆 Final F1-Score: {results['f1_score']:.4f}")
    
    # Performance assessment
    if results['f1_score'] > 0.85:
        print("✅ Excellent performance - SVM works great for this task!")
    elif results['f1_score'] > 0.75:
        print("⚠️  Good performance - Consider parameter tuning")
    else:
        print("❌ Poor performance - SVM may not be suitable")
    
    # Save results
    classifier.save_results("svm_results.json")
    
    # Feature analysis (for linear kernel)
    classifier.get_feature_importance()
    
    print(f"\n📊 Results Summary:")
    print(f"   Model: SVM")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")
    if results['auc']:
        print(f"   AUC: {results['auc']:.4f}")
    
    return results

if __name__ == "__main__":
    main()