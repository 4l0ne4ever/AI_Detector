#!/usr/bin/env python3
"""
Model Evaluation and Analysis
============================

This script provides detailed evaluation and visualization of trained models.
Includes ROC curves, confusion matrices, feature importance, and error analysis.

Usage:
    python3 evaluate_models.py
"""

import json
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, models_dir="../trained_models", results_dir="../model_results", data_dir="../processed_data"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.data_dir = data_dir
        
        # Create evaluation directory
        self.eval_dir = f"{results_dir}/detailed_evaluation"
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Data
        self.test_data = None
        self.models = {}
        
    def load_data_and_models(self):
        """Load test data and trained models"""
        
        print("Loading test data and models...")
        
        # Load test data
        try:
            self.test_data = pd.read_json(f"{self.data_dir}/test_data.jsonl", lines=True)
            self.test_data['label_binary'] = self.test_data['label'].map({'human': 0, 'ai': 1})
            print(f"Loaded test data: {len(self.test_data):,} samples")
        except FileNotFoundError:
            print("Test data not found. Run preprocessing.py first!")
            return False
        
        # Load trained models
        model_files = {
            'SVM': 'svm_model.pkl',
            'Random_Forest': 'random_forest_model.pkl', 
            'Naive_Bayes': 'naive_bayes_model.pkl',
            'Logistic_Regression': 'logistic_regression_model.pkl'
        }
        
        for name, filename in model_files.items():
            try:
                with open(f"{self.models_dir}/{filename}", 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded {name} model")
            except FileNotFoundError:
                print(f"{name} model not found, skipping...")
        
        if not self.models:
            print("No trained models found. Run train_models.py first!")
            return False
            
        return True
    
    def create_confusion_matrices(self):
        """Create confusion matrix plots for all models"""
        
        print("\\nCreating confusion matrices...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        X_test = self.test_data['text']
        y_test = self.test_data['label_binary']
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= 4:  # Max 4 models in 2x2 grid
                break
                
            # Predictions
            y_pred = model.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'],
                ax=axes[idx]
            )
            axes[idx].set_title(f'{name}\\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(len(self.models), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        print(f"   Confusion matrices saved")
        plt.close()
    
    def create_roc_curves(self):
        """Create ROC curve comparison plot"""
        
        print("\\nCreating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        X_test = self.test_data['text']
        y_test = self.test_data['label_binary']
        
        for name, model in self.models.items():
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Human vs AI Text Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{self.eval_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        print(f"   ROC curves saved")
        plt.close()
    
    def create_precision_recall_curves(self):
        """Create Precision-Recall curve comparison plot"""
        
        print("\\nCreating Precision-Recall curves...")
        
        plt.figure(figsize=(10, 8))
        
        X_test = self.test_data['text']
        y_test = self.test_data['label_binary']
        
        for name, model in self.models.items():
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Human vs AI Text Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{self.eval_dir}/precision_recall_curves.png", dpi=300, bbox_inches='tight')
        print(f"   Precision-Recall curves saved")
        plt.close()
    
    def analyze_feature_importance(self):
        """Analyze feature importance for applicable models"""
        
        print("\\nAnalyzing feature importance...")
        
        # Only for models with feature importance
        importance_models = ['Random_Forest', 'Logistic_Regression']
        
        for name in importance_models:
            if name not in self.models:
                continue
                
            model = self.models[name]
            
            try:
                # Get feature names (TF-IDF terms)
                feature_names = model.named_steps['tfidf'].get_feature_names_out()
                
                # Get importance scores
                if name == 'Random_Forest':
                    importance = model.named_steps['rf'].feature_importances_
                elif name == 'Logistic_Regression':
                    importance = np.abs(model.named_steps['lr'].coef_[0])
                
                # Top features
                top_indices = importance.argsort()[-20:][::-1]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = importance[top_indices]
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(top_features)), top_importance)
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('Importance Score')
                plt.title(f'{name} - Top 20 Most Important Features')
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                plt.savefig(f"{self.eval_dir}/{name.lower()}_feature_importance.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   {name} feature importance saved")
                
                # Save top features to text file
                with open(f"{self.eval_dir}/{name.lower()}_top_features.txt", 'w') as f:
                    f.write(f"Top 20 Features for {name}\\n")
                    f.write("="*40 + "\\n")
                    for i, (feature, score) in enumerate(zip(top_features, top_importance)):
                        f.write(f"{i+1:2d}. {feature:<20} {score:.6f}\\n")
                        
            except Exception as e:
                print(f"   Could not analyze {name} features: {e}")
    
    def perform_error_analysis(self):
        """Analyze misclassified examples"""
        
        print("\\nPerforming error analysis...")
        
        X_test = self.test_data['text']
        y_test = self.test_data['label_binary']
        
        # Use best performing model (assume it's SVM for now)
        best_model_name = 'SVM'  # This could be determined from results
        if best_model_name not in self.models:
            best_model_name = list(self.models.keys())[0]
        
        model = self.models[best_model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Find misclassified examples
        misclassified_mask = y_test != y_pred
        misclassified_data = self.test_data[misclassified_mask].copy()
        
        if len(misclassified_data) == 0:
            print("   No misclassifications found!")
            return
        
        # Add prediction info
        misclassified_data['predicted'] = y_pred[misclassified_mask]
        misclassified_data['confidence'] = np.max(y_pred_proba[misclassified_mask], axis=1)
        
        # Separate false positives and false negatives
        false_positives = misclassified_data[
            (misclassified_data['label'] == 'human') & 
            (misclassified_data['predicted'] == 1)
        ]
        
        false_negatives = misclassified_data[
            (misclassified_data['label'] == 'ai') & 
            (misclassified_data['predicted'] == 0)
        ]
        
        print(f"   Error Analysis Results:")
        print(f"      Total misclassified: {len(misclassified_data):,}")
        print(f"      False positives (Human→AI): {len(false_positives):,}")
        print(f"      False negatives (AI→Human): {len(false_negatives):,}")
        
        # Save detailed error analysis
        error_analysis = {
            'model_used': best_model_name,
            'total_test_samples': len(self.test_data),
            'total_misclassified': len(misclassified_data),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'error_rate': len(misclassified_data) / len(self.test_data)
        }
        
        with open(f"{self.eval_dir}/error_analysis.json", 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Save examples of misclassified texts
        if len(false_positives) > 0:
            fp_sample = false_positives.head(10)[['text', 'confidence']].to_dict('records')
            with open(f"{self.eval_dir}/false_positives_examples.json", 'w') as f:
                json.dump(fp_sample, f, indent=2)
        
        if len(false_negatives) > 0:
            fn_sample = false_negatives.head(10)[['text', 'confidence']].to_dict('records')
            with open(f"{self.eval_dir}/false_negatives_examples.json", 'w') as f:
                json.dump(fn_sample, f, indent=2)
        
        print(f"   Error analysis saved to {self.eval_dir}/")
    
    def create_performance_comparison(self):
        """Create performance comparison visualization"""
        
        print("\\nCreating performance comparison...")
        
        # Load results from training
        try:
            with open(f"{self.results_dir}/training_results.json", 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            print("   Training results not found, skipping comparison")
            return
        
        # Extract metrics for plotting
        models = list(results.keys())
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            
            bars = axes[idx].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            axes[idx].set_title(f'{metric.replace("test_", "").replace("_", " ").title()}')
            axes[idx].set_ylim(0, 1.1)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        train_times = [results[model].get('train_time', 0) for model in models]
        bars = axes[5].bar(models, train_times, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[5].set_title('Training Time (seconds)')
        axes[5].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars, train_times):
            axes[5].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_times)*0.01,
                        f'{time:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        print(f"   Performance comparison saved")
        plt.close()
    
    def generate_detailed_report(self):
        """Generate a comprehensive evaluation report"""
        
        print("\\nGenerating detailed evaluation report...")
        
        X_test = self.test_data['text']
        y_test = self.test_data['label_binary']
        
        report_content = []
        report_content.append("# Human vs AI Text Classification - Detailed Evaluation Report")
        report_content.append("=" * 70)
        report_content.append(f"\\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Test samples: {len(self.test_data):,}")
        report_content.append("\\n")
        
        # Model performance summary
        report_content.append("## Model Performance Summary")
        report_content.append("-" * 30)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            report_content.append(f"\\n### {name}")
            report_content.append(f"- Accuracy:  {accuracy:.4f}")
            report_content.append(f"- Precision: {precision:.4f}")
            report_content.append(f"- Recall:    {recall:.4f}")
            report_content.append(f"- F1-Score:  {f1:.4f}")
            report_content.append(f"- AUC-ROC:   {auc:.4f}")
            
            # Classification report
            report_content.append(f"\\n#### Detailed Classification Report:")
            report_content.append("```")
            report_content.append(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
            report_content.append("```")
        
        # Save report
        with open(f"{self.eval_dir}/evaluation_report.md", 'w') as f:
            f.write("\\n".join(report_content))
        
        print(f"   Detailed report saved")

def main():
    """Main evaluation pipeline"""
    
    print("Human vs AI Text Classification - Model Evaluation")
    print("="*55)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load data and models
    if not evaluator.load_data_and_models():
        return
    
    # Run all evaluations
    evaluator.create_confusion_matrices()
    evaluator.create_roc_curves()  
    evaluator.create_precision_recall_curves()
    evaluator.analyze_feature_importance()
    evaluator.perform_error_analysis()
    evaluator.create_performance_comparison()
    evaluator.generate_detailed_report()
    
    print("\\nModel evaluation completed!")
    print(f"All results saved to: model_results/detailed_evaluation/")
    print("\\nFiles created:")
    print("   - confusion_matrices.png")
    print("   - roc_curves.png") 
    print("   - precision_recall_curves.png")
    print("   - feature_importance plots")
    print("   - error_analysis.json")
    print("   - performance_comparison.png")
    print("   - evaluation_report.md")

if __name__ == "__main__":
    main()