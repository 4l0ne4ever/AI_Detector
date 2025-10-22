#!/usr/bin/env python3
"""
MLP (Multi-Layer Perceptron) Classifier for AI vs Human Text Detection
Uses shared configuration and data preprocessing
"""

import json
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from config import (
    DATA_CONFIG, TRAINING_CONFIG, MODEL_PARAMS, 
    data_loader, feature_extractor, evaluator
)

class MLPTextClassifier:
    """MLP-based AI text classifier"""
    
    def __init__(self):
        self.model = None
        self.results = None
        
    def create_model(self):
        """Create MLP model with shared parameters"""
        params = MODEL_PARAMS['mlp']
        self.model = MLPClassifier(**params)
        print(f"MLP Model created with parameters:")
        print(f"  Hidden layers: {params['hidden_layer_sizes']}")
        print(f"  Activation: {params['activation']}")
        print(f"  Solver: {params['solver']}")
        print(f"  Learning rate: {params['learning_rate_init']}")
        print(f"  Max iterations: {params['max_iter']}")
        return self.model
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline"""
        print("🧠 MLP Neural Network Classifier for AI vs Human Text Detection")
        print("=" * 70)
        
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
        
        print(f"\n🔥 Training Neural Network...")
        print(f"Note: MLP may take longer than other models due to iterative optimization")
        
        self.results = evaluator.evaluate_model(
            self.model, X_train, X_test, y_train, y_test, "MLP"
        )
        
        # Additional MLP-specific info
        if hasattr(self.model, 'n_iter_'):
            print(f"Training converged after {self.model.n_iter_} iterations")
        
        if hasattr(self.model, 'loss_'):
            print(f"Final training loss: {self.model.loss_:.6f}")
            
        return self.results
    
    def save_results(self, filename: str = "mlp_results.json"):
        """Save results to file"""
        if self.results:
            evaluator.save_results(self.results, filename)
        else:
            print("No results to save. Run train_and_evaluate() first.")
    
    def analyze_network(self):
        """Analyze the trained neural network"""
        if not self.model:
            print("Model must be trained first")
            return None
            
        try:
            print(f"\n🔍 Neural Network Analysis:")
            print(f"Input layer size: {self.model.coefs_[0].shape[0]}")
            
            for i, layer in enumerate(self.model.coefs_):
                print(f"Layer {i+1}: {layer.shape[0]} → {layer.shape[1]} neurons")
            
            print(f"Output layer size: {self.model.coefs_[-1].shape[1]}")
            
            # Weight statistics
            total_weights = sum(layer.size for layer in self.model.coefs_)
            print(f"Total parameters: {total_weights:,}")
            
            # Layer-wise statistics
            for i, (weights, biases) in enumerate(zip(self.model.coefs_, self.model.intercepts_)):
                print(f"\nLayer {i+1} statistics:")
                print(f"  Weights mean: {np.mean(weights):.6f}")
                print(f"  Weights std: {np.std(weights):.6f}")
                print(f"  Biases mean: {np.mean(biases):.6f}")
                
            return {
                'architecture': [layer.shape for layer in self.model.coefs_],
                'total_parameters': total_weights,
                'iterations': getattr(self.model, 'n_iter_', 'Unknown'),
                'final_loss': getattr(self.model, 'loss_', 'Unknown')
            }
            
        except Exception as e:
            print(f"Error analyzing network: {e}")
            return None
    
    def get_prediction_confidence(self, texts, top_k=5):
        """Analyze prediction confidence for sample texts"""
        if not self.model or not feature_extractor.is_fitted:
            print("Model must be trained first")
            return None
            
        try:
            features = feature_extractor.transform(texts)
            probabilities = self.model.predict_proba(features)
            predictions = self.model.predict(features)
            
            print(f"\n📊 Prediction Confidence Analysis (top {min(top_k, len(texts))}):")
            
            for i in range(min(top_k, len(texts))):
                human_prob = probabilities[i][1] if self.model.classes_[1] == 'human' else probabilities[i][0]
                ai_prob = 1 - human_prob
                confidence = max(human_prob, ai_prob)
                
                print(f"\nSample {i+1}:")
                print(f"  Text: {texts[i][:100]}...")
                print(f"  Prediction: {predictions[i]}")
                print(f"  Human probability: {human_prob:.4f}")
                print(f"  AI probability: {ai_prob:.4f}")
                print(f"  Confidence: {confidence:.4f}")
                
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'confidences': [max(prob) for prob in probabilities]
            }
            
        except Exception as e:
            print(f"Error analyzing predictions: {e}")
            return None

def main():
    """Main execution for MLP classifier"""
    classifier = MLPTextClassifier()
    
    # Train and evaluate
    start_time = time.time()
    results = classifier.train_and_evaluate()
    total_time = time.time() - start_time
    
    print(f"\n🎯 MLP Classification Complete!")
    print(f"⚡ Total execution time: {total_time:.1f} seconds")
    print(f"🏆 Final F1-Score: {results['f1_score']:.4f}")
    
    # Performance assessment
    if results['f1_score'] > 0.85:
        print("✅ Excellent performance - MLP neural network works great!")
    elif results['f1_score'] > 0.75:
        print("⚠️  Good performance - Consider architecture tuning")
    else:
        print("❌ Poor performance - Try different architecture or features")
    
    # Save results
    classifier.save_results("mlp_results.json")
    
    # Network analysis
    classifier.analyze_network()
    
    print(f"\n📊 Results Summary:")
    print(f"   Model: MLP Neural Network")
    print(f"   Architecture: {MODEL_PARAMS['mlp']['hidden_layer_sizes']}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")
    if results['auc']:
        print(f"   AUC: {results['auc']:.4f}")
    
    print(f"\n🧠 Neural Network Advantages:")
    print(f"   ✓ Can learn complex non-linear patterns")
    print(f"   ✓ Automatic feature combination")
    print(f"   ✓ Scalable to larger datasets")
    print(f"   ✓ Provides prediction probabilities")
    
    return results

if __name__ == "__main__":
    main()