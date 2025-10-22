#!/usr/bin/env python3
"""
Human vs AI Text Classifier - Prediction Interface
=================================================

This script provides an easy-to-use interface for predicting whether
a given text is human-written or AI-generated.

Usage:
    python3 predict_text.py
    python3 predict_text.py --text "Your text here"
    python3 predict_text.py --file "path/to/text/file.txt"
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

class HumanAIClassifier:
    def __init__(self, models_dir="../trained_models", results_dir="../model_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.model = None
        self.model_name = None
        
    def load_best_model(self):
        """Load the best performing model"""
        
        print("Loading best model...")
        
        # Try to load results to find best model
        try:
            with open(f"{self.results_dir}/training_results.json", 'r') as f:
                results = json.load(f)
            
            # Find best model based on F1 score
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x].get('test_f1', 0))
            
        except FileNotFoundError:
            print("Training results not found, using SVM as default")
            best_model_name = 'SVM'
        
        # Model filename mapping
        model_files = {
            'SVM': 'svm_model.pkl',
            'Random_Forest': 'random_forest_model.pkl',
            'Naive_Bayes': 'naive_bayes_model.pkl', 
            'Logistic_Regression': 'logistic_regression_model.pkl'
        }
        
        # Try to load the best model
        model_file = model_files.get(best_model_name)
        if model_file and os.path.exists(f"{self.models_dir}/{model_file}"):
            try:
                with open(f"{self.models_dir}/{model_file}", 'rb') as f:
                    self.model = pickle.load(f)
                self.model_name = best_model_name
                print(f"Loaded {best_model_name} model")
                return True
            except Exception as e:
                print(f"Error loading {best_model_name}: {e}")
        
        # Fallback: try to load any available model
        for name, filename in model_files.items():
            if os.path.exists(f"{self.models_dir}/{filename}"):
                try:
                    with open(f"{self.models_dir}/{filename}", 'rb') as f:
                        self.model = pickle.load(f)
                    self.model_name = name
                    print(f"Loaded {name} model (fallback)")
                    return True
                except Exception:
                    continue
        
        print("No trained models found!")
        print("Run train_models.py first to train the models")
        return False
    
    def clean_text(self, text):
        """Clean input text similar to training preprocessing"""
        if not isinstance(text, str):
            return ""
        
        import re
        
        # Basic cleaning
        text = re.sub(r'\\s+', ' ', text).strip()
        text = re.sub(r'[^\\w\\s\\.,;:!?\\-\\(\\)]', ' ', text)
        text = re.sub(r'\\s+', ' ', text)
        
        return text.strip()
    
    def predict_single_text(self, text):
        """Predict if a single text is human or AI-written"""
        
        if not self.model:
            print("No model loaded!")
            return None
        
        # Clean the text
        clean_text = self.clean_text(text)
        
        if len(clean_text) < 10:
            print("Text too short for reliable prediction")
            return None
        
        try:
            # Make prediction
            prediction = self.model.predict([clean_text])[0]
            probabilities = self.model.predict_proba([clean_text])[0]
            
            # Convert to human-readable format
            human_prob = probabilities[0]
            ai_prob = probabilities[1]
            
            predicted_label = "Human" if prediction == 0 else "AI"
            confidence = max(human_prob, ai_prob)
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'human_probability': human_prob,
                'ai_probability': ai_prob,
                'model_used': self.model_name,
                'text_length': len(clean_text),
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def predict_batch(self, texts):
        """Predict multiple texts at once"""
        
        if not self.model:
            print("No model loaded!")
            return None
        
        results = []
        for i, text in enumerate(texts):
            result = self.predict_single_text(text)
            if result:
                result['text_id'] = i
                results.append(result)
        
        return results
    
    def format_prediction_output(self, result):
        """Format prediction result for display"""
        
        if not result:
            return "Prediction failed"
        
        output = []
        output.append("PREDICTION RESULT")
        output.append("=" * 20)
        output.append(f"Predicted: {result['predicted_label']}")
        output.append(f"Confidence: {result['confidence']:.1%}")
        output.append("")
        output.append("Detailed Probabilities:")
        output.append(f"   Human: {result['human_probability']:.1%}")
        output.append(f"   AI:    {result['ai_probability']:.1%}")
        output.append("")
        output.append(f"Model: {result['model_used']}")
        output.append(f"Text length: {result['text_length']:,} characters")
        
        # Confidence interpretation
        if result['confidence'] >= 0.9:
            confidence_desc = "Very High"
        elif result['confidence'] >= 0.8:
            confidence_desc = "High"
        elif result['confidence'] >= 0.7:
            confidence_desc = "Moderate"
        else:
            confidence_desc = "Low"
        
        output.append(f"Confidence level: {confidence_desc}")
        
        return "\\n".join(output)

def interactive_mode(classifier):
    """Interactive prediction mode"""
    
    print("\\nInteractive Human vs AI Text Classification")
    print("=" * 45)
    print("Enter text to classify (or 'quit' to exit)")
    print("Tip: Paste scientific abstracts for best results\\n")
    
    while True:
        try:
            print("Enter text to classify:")
            text = input("> ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if len(text.strip()) < 10:
                print("Please enter more text (at least 10 characters)\\n")
                continue
            
            print("\\nAnalyzing...")
            result = classifier.predict_single_text(text)
            
            if result:
                print(classifier.format_prediction_output(result))
            else:
                print("Could not analyze this text")
            
            print("\\n" + "-" * 50 + "\\n")
            
        except KeyboardInterrupt:
            print("\\n\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\\n")

def main():
    """Main prediction interface"""
    
    parser = argparse.ArgumentParser(
        description="Classify text as human-written or AI-generated"
    )
    parser.add_argument(
        '--text', 
        type=str, 
        help='Text to classify'
    )
    parser.add_argument(
        '--file', 
        type=str, 
        help='Path to text file to classify'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple texts from file (one per line)'
    )
    
    args = parser.parse_args()
    
    print("Human vs AI Text Classifier")
    print("=" * 35)
    
    # Initialize classifier
    classifier = HumanAIClassifier()
    
    # Load best model
    if not classifier.load_best_model():
        return
    
    # Command line text input
    if args.text:
        print(f"\\nAnalyzing provided text...")
        result = classifier.predict_single_text(args.text)
        
        if result:
            print("\\n" + classifier.format_prediction_output(result))
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\\nResults saved to {args.output}")
        else:
            print("Could not analyze the provided text")
    
    # File input
    elif args.file:
        try:
            print(f"\\nReading from {args.file}...")
            
            with open(args.file, 'r', encoding='utf-8') as f:
                if args.batch:
                    # Multiple texts, one per line
                    texts = [line.strip() for line in f if line.strip()]
                    print(f"Processing {len(texts)} texts...")
                    
                    results = classifier.predict_batch(texts)
                    
                    if results:
                        print(f"\\nProcessed {len(results)} texts successfully\\n")
                        
                        # Summary
                        human_count = sum(1 for r in results if r['predicted_label'] == 'Human')
                        ai_count = len(results) - human_count
                        
                        print("BATCH RESULTS SUMMARY")
                        print("=" * 25)
                        print(f"Human texts: {human_count}")
                        print(f"AI texts: {ai_count}")
                        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.1%}")
                        
                        # Save results if requested
                        if args.output:
                            with open(args.output, 'w') as outf:
                                json.dump(results, outf, indent=2)
                            print(f"\\nDetailed results saved to {args.output}")
                    
                else:
                    # Single text from file
                    text = f.read()
                    result = classifier.predict_single_text(text)
                    
                    if result:
                        print("\\n" + classifier.format_prediction_output(result))
                        
                        if args.output:
                            with open(args.output, 'w') as outf:
                                json.dump(result, outf, indent=2)
                            print(f"\\nResults saved to {args.output}")
                    
        except FileNotFoundError:
            print(f"File not found: {args.file}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Interactive mode
    else:
        interactive_mode(classifier)

if __name__ == "__main__":
    main()