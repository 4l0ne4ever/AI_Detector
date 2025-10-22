#!/usr/bin/env python3
"""
Baseline Classifiers for AI vs Human Text Detection
Includes Naive Bayes and Logistic Regression
Uses shared configuration and data preprocessing
"""

import json
import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from config import (
    DATA_CONFIG, TRAINING_CONFIG, MODEL_PARAMS, 
    data_loader, feature_extractor, evaluator
)

class BaselineClassifiers:
    """Baseline classifiers for comparison"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def create_naive_bayes(self):
        """Create Naive Bayes model"""
        params = MODEL_PARAMS['naive_bayes']
        model = MultinomialNB(**params)
        print(f"Naive Bayes created with parameters: {params}")
        return model
    
    def create_logistic_regression(self):
        """Create Logistic Regression model"""
        params = MODEL_PARAMS['logistic_regression']
        model = LogisticRegression(**params)
        print(f"Logistic Regression created with parameters: {params}")
        return model
    
    def train_and_evaluate_all(self):
        """Train and evaluate both baseline models"""
        print("📊 Baseline Classifiers for AI vs Human Text Detection")
        print("=" * 65)
        
        # Load data using shared loader
        texts, labels = data_loader.create_balanced_dataset(
            DATA_CONFIG['samples_per_class']
        )
        
        # Split data (same split for both models)
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
        
        # Train and evaluate both models
        models_to_test = {
            'Naive Bayes': self.create_naive_bayes(),
            'Logistic Regression': self.create_logistic_regression()
        }
        
        all_results = []
        
        for name, model in models_to_test.items():
            print(f"\n{'='*50}")
            self.models[name] = model
            
            results = evaluator.evaluate_model(
                model, X_train, X_test, y_train, y_test, name
            )
            
            self.results[name] = results
            all_results.append(results)
        
        return all_results
    
    def compare_models(self):
        """Compare performance of baseline models"""
        if not self.results:
            print("No results to compare. Run train_and_evaluate_all() first.")
            return
            
        print(f"\n" + "="*80)
        print("📈 BASELINE MODEL COMPARISON")
        print("="*80)
        
        # Create comparison table
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Time':<8}")
        print("-" * 80)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for name, result in sorted_results:
            auc_str = f"{result['auc']:.4f}" if result['auc'] else "N/A"
            print(f"{name:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} {result['f1_score']:<10.4f} {auc_str:<10} "
                  f"{result['train_time']:<8.2f}s")
        
        # Best model
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 Best Baseline Model: {best_name}")
        print(f"   F1-Score: {best_result['f1_score']:.4f}")
        print(f"   Cross-validation: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")
        
        # Speed comparison
        fastest = min(self.results.items(), key=lambda x: x[1]['train_time'])
        print(f"\n⚡ Fastest Model: {fastest[0]} ({fastest[1]['train_time']:.2f}s)")
        
        return sorted_results
    
    def analyze_features(self, model_name='Logistic Regression', top_k=15):
        """Analyze important features for interpretable models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
            
        model = self.models[model_name]
        
        if not feature_extractor.is_fitted:
            print("Models must be trained first")
            return None
            
        try:
            if hasattr(model, 'coef_') and model.coef_ is not None:
                feature_names = feature_extractor.tfidf_vectorizer.get_feature_names_out()
                coef = model.coef_[0]
                
                # Get top positive and negative coefficients
                top_positive_idx = coef.argsort()[-top_k:][::-1]
                top_negative_idx = coef.argsort()[:top_k]
                
                print(f"\n🔍 Feature Analysis for {model_name}:")
                print(f"\nTop {top_k} features indicating HUMAN text:")
                for idx in top_positive_idx:
                    print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
                
                print(f"\nTop {top_k} features indicating AI text:")
                for idx in top_negative_idx:
                    print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
                
                return {
                    'human_features': [(feature_names[idx], float(coef[idx])) for idx in top_positive_idx],
                    'ai_features': [(feature_names[idx], float(coef[idx])) for idx in top_negative_idx]
                }
                
            elif hasattr(model, 'feature_log_prob_'):
                # For Naive Bayes
                feature_names = feature_extractor.tfidf_vectorizer.get_feature_names_out()
                
                # Get class indices
                human_idx = list(model.classes_).index('human')
                ai_idx = list(model.classes_).index('ai')
                
                # Calculate log probability differences
                log_prob_diff = model.feature_log_prob_[human_idx] - model.feature_log_prob_[ai_idx]
                
                # Top features for each class
                top_human_idx = log_prob_diff.argsort()[-top_k:][::-1]
                top_ai_idx = log_prob_diff.argsort()[:top_k]
                
                print(f"\n🔍 Feature Analysis for {model_name}:")
                print(f"\nTop {top_k} features indicating HUMAN text:")
                for idx in top_human_idx:
                    print(f"  {feature_names[idx]}: {log_prob_diff[idx]:.4f}")
                
                print(f"\nTop {top_k} features indicating AI text:")
                for idx in top_ai_idx:
                    print(f"  {feature_names[idx]}: {log_prob_diff[idx]:.4f}")
                
                return {
                    'human_features': [(feature_names[idx], float(log_prob_diff[idx])) for idx in top_human_idx],
                    'ai_features': [(feature_names[idx], float(log_prob_diff[idx])) for idx in top_ai_idx]
                }
            else:
                print(f"Feature analysis not available for {model_name}")
                return None
                
        except Exception as e:
            print(f"Error analyzing features: {e}")
            return None
    
    def save_all_results(self):
        """Save all results to files"""
        for name, results in self.results.items():
            filename = f"{name.lower().replace(' ', '_')}_results.json"
            evaluator.save_results(results, filename)

def main():
    """Main execution for baseline classifiers"""
    classifiers = BaselineClassifiers()
    
    print("🚀 Starting baseline model comparison...")
    
    # Train and evaluate
    start_time = time.time()
    all_results = classifiers.train_and_evaluate_all()
    total_time = time.time() - start_time
    
    print(f"\n🎯 Baseline Classification Complete!")
    print(f"⚡ Total execution time: {total_time:.1f} seconds")
    
    # Compare models
    comparison = classifiers.compare_models()
    
    # Performance assessment
    best_name, best_result = comparison[0]
    print(f"\n📈 Performance Assessment:")
    if best_result['f1_score'] > 0.85:
        print("✅ Excellent baseline performance!")
    elif best_result['f1_score'] > 0.75:
        print("⚠️  Good baseline performance")
    else:
        print("❌ Poor baseline performance - may need better features")
    
    # Save results
    classifiers.save_all_results()
    
    # Feature analysis
    print(f"\n" + "="*60)
    classifiers.analyze_features('Logistic Regression', 10)
    classifiers.analyze_features('Naive Bayes', 10)
    
    print(f"\n💡 Baseline Model Insights:")
    print(f"   📊 Naive Bayes: Fast, good for text, assumes feature independence")
    print(f"   🎯 Logistic Regression: Interpretable, handles correlated features")
    print(f"   ⚡ Both models are fast to train and predict")
    print(f"   🔍 Feature analysis shows what distinguishes AI from human text")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Compare these results with SVM and MLP")
    print(f"   2. If baselines perform well (F1 > 0.8), the problem is feasible")
    print(f"   3. If baselines are poor, focus on better feature engineering")
    print(f"   4. Use feature analysis to understand text differences")
    
    return classifiers.results

if __name__ == "__main__":
    main()