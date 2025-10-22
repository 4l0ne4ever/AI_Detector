#!/usr/bin/env python3
"""
Complete Model Comparison Script
Runs SVM, MLP, and Baseline models, then compares results
"""

import json
import time
import pandas as pd
from pathlib import Path

# Import all classifiers
from svm_classifier import SVMClassifier
from mlp_classifier import MLPTextClassifier
from baseline_classifiers import BaselineClassifiers

def run_all_models():
    """Run all classification models"""
    print("🚀 COMPREHENSIVE AI vs HUMAN TEXT CLASSIFICATION COMPARISON")
    print("=" * 80)
    
    all_results = []
    execution_times = {}
    
    # 1. Baseline Models
    print(f"\n{'='*20} BASELINE MODELS {'='*20}")
    start_time = time.time()
    baseline_classifier = BaselineClassifiers()
    baseline_results = baseline_classifier.train_and_evaluate_all()
    baseline_time = time.time() - start_time
    
    # Add baseline results
    for result in baseline_results:
        all_results.append(result)
    execution_times['Baselines'] = baseline_time
    
    # 2. SVM Model
    print(f"\n{'='*25} SVM MODEL {'='*25}")
    start_time = time.time()
    svm_classifier = SVMClassifier()
    svm_result = svm_classifier.train_and_evaluate()
    svm_time = time.time() - start_time
    
    all_results.append(svm_result)
    execution_times['SVM'] = svm_time
    
    # 3. MLP Model
    print(f"\n{'='*25} MLP MODEL {'='*25}")
    start_time = time.time()
    mlp_classifier = MLPTextClassifier()
    mlp_result = mlp_classifier.train_and_evaluate()
    mlp_time = time.time() - start_time
    
    all_results.append(mlp_result)
    execution_times['MLP'] = mlp_time
    
    return {
        'results': all_results,
        'execution_times': execution_times,
        'classifiers': {
            'baseline': baseline_classifier,
            'svm': svm_classifier,
            'mlp': mlp_classifier
        }
    }

def create_comparison_table(results):
    """Create comprehensive comparison table"""
    print(f"\n" + "="*100)
    print("🏆 FINAL MODEL COMPARISON")
    print("="*100)
    
    # Create DataFrame for easy comparison
    df_data = []
    for result in results:
        df_data.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'AUC': result['auc'] if result['auc'] else 0,
            'CV Mean': result['cv_mean'],
            'CV Std': result['cv_std'],
            'Train Time': result['train_time'],
            'Predict Time': result['predict_time']
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    # Print formatted table
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Train(s)':<10} {'CV Score':<12}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        auc_str = f"{row['AUC']:.4f}" if row['AUC'] > 0 else "N/A"
        cv_str = f"{row['CV Mean']:.4f}±{row['CV Std']:.3f}"
        print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} "
              f"{row['Recall']:<10.4f} {row['F1-Score']:<10.4f} {auc_str:<10} "
              f"{row['Train Time']:<10.2f} {cv_str:<12}")
    
    return df

def analyze_results(df, execution_times):
    """Analyze and provide insights"""
    best_model = df.iloc[0]
    
    print(f"\n" + "="*60)
    print("📊 ANALYSIS & INSIGHTS")
    print("="*60)
    
    print(f"\n🏆 BEST PERFORMING MODEL:")
    print(f"   Model: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    if best_model['AUC'] > 0:
        print(f"   AUC: {best_model['AUC']:.4f}")
    print(f"   Cross-validation: {best_model['CV Mean']:.4f} (±{best_model['CV Std']:.4f})")
    
    # Performance assessment
    print(f"\n📈 PERFORMANCE ASSESSMENT:")
    if best_model['F1-Score'] > 0.85:
        print("   ✅ EXCELLENT - Ready for production scaling!")
        recommendation = "PROCEED"
    elif best_model['F1-Score'] > 0.75:
        print("   ⚠️  GOOD - Consider feature engineering or ensemble methods")
        recommendation = "IMPROVE"
    else:
        print("   ❌ POOR - Need better approach (deep learning, better features)")
        recommendation = "REDESIGN"
    
    # Speed analysis
    print(f"\n⚡ SPEED ANALYSIS:")
    fastest_idx = df['Train Time'].idxmin()
    fastest_model = df.iloc[fastest_idx]
    slowest_idx = df['Train Time'].idxmax()
    slowest_model = df.iloc[slowest_idx]
    
    print(f"   Fastest: {fastest_model['Model']} ({fastest_model['Train Time']:.2f}s)")
    print(f"   Slowest: {slowest_model['Model']} ({slowest_model['Train Time']:.2f}s)")
    print(f"   Speed ratio: {slowest_model['Train Time'] / fastest_model['Train Time']:.1f}x")
    
    # Robustness analysis
    print(f"\n🎯 ROBUSTNESS ANALYSIS:")
    most_stable_idx = df['CV Std'].idxmin()
    most_stable = df.iloc[most_stable_idx]
    print(f"   Most stable: {most_stable['Model']} (CV std: {most_stable['CV Std']:.4f})")
    
    # Model-specific insights
    print(f"\n💡 MODEL-SPECIFIC INSIGHTS:")
    for _, row in df.iterrows():
        model_name = row['Model']
        if 'SVM' in model_name:
            print(f"   🤖 SVM: Good balance of performance and interpretability")
        elif 'MLP' in model_name:
            print(f"   🧠 MLP: Can capture complex patterns, may need more data to shine")
        elif 'Naive Bayes' in model_name:
            print(f"   📊 Naive Bayes: Fast baseline, assumes feature independence")
        elif 'Logistic' in model_name:
            print(f"   🎯 Logistic Regression: Highly interpretable, good baseline")
    
    return {
        'recommendation': recommendation,
        'best_model': best_model['Model'],
        'best_f1': best_model['F1-Score'],
        'fastest_model': fastest_model['Model'],
        'most_stable': most_stable['Model']
    }

def save_comprehensive_results(all_data, analysis):
    """Save comprehensive results to JSON"""
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_info': {
            'samples_per_class': 1000,  # From config
            'total_samples': 2000,
            'test_split': 0.2
        },
        'results': all_data['results'],
        'execution_times': all_data['execution_times'],
        'analysis': analysis,
        'recommendations': {
            'best_model': analysis['best_model'],
            'recommendation': analysis['recommendation'],
            'next_steps': get_next_steps(analysis)
        }
    }
    
    with open('comprehensive_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Comprehensive results saved to: comprehensive_comparison_results.json")

def get_next_steps(analysis):
    """Generate next steps based on analysis"""
    if analysis['recommendation'] == 'PROCEED':
        return [
            "Scale up dataset to 10K-50K samples",
            f"Deploy {analysis['best_model']} for production testing",
            "Consider ensemble methods for even better performance",
            "Implement real-time classification API"
        ]
    elif analysis['recommendation'] == 'IMPROVE':
        return [
            "Try feature engineering (n-grams, linguistic features)",
            "Experiment with ensemble methods",
            "Increase dataset size gradually",
            f"Fine-tune {analysis['best_model']} hyperparameters"
        ]
    else:
        return [
            "Try deep learning approaches (BERT, RoBERTa)",
            "Improve AI text generation quality",
            "Add more sophisticated features",
            "Consider domain-specific models"
        ]

def print_final_recommendation(analysis):
    """Print final recommendation"""
    print(f"\n" + "="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)
    
    if analysis['recommendation'] == 'PROCEED':
        print("✅ PROJECT IS READY TO SCALE!")
        print(f"   Your best model ({analysis['best_model']}) achieved F1-Score: {analysis['best_f1']:.4f}")
        print("   This indicates the AI vs Human classification problem is solvable.")
        print("   You can confidently invest in scaling up the dataset and deployment.")
        
    elif analysis['recommendation'] == 'IMPROVE':
        print("⚠️  PROJECT SHOWS PROMISE - NEEDS IMPROVEMENT")
        print(f"   Your best model ({analysis['best_model']}) achieved F1-Score: {analysis['best_f1']:.4f}")
        print("   The approach is viable but needs optimization before scaling.")
        print("   Focus on feature engineering and hyperparameter tuning.")
        
    else:
        print("❌ PROJECT NEEDS FUNDAMENTAL CHANGES")
        print(f"   Best F1-Score: {analysis['best_f1']:.4f} is too low for production")
        print("   Consider switching to deep learning approaches or improving data quality.")
        print("   This prototype helped you avoid a larger investment in a poor approach.")

def main():
    """Main execution"""
    print("Starting comprehensive model comparison...")
    
    total_start_time = time.time()
    
    # Run all models
    all_data = run_all_models()
    
    # Create comparison
    df = create_comparison_table(all_data['results'])
    
    # Analyze results
    analysis = analyze_results(df, all_data['execution_times'])
    
    # Save results
    save_comprehensive_results(all_data, analysis)
    
    # Final recommendation
    print_final_recommendation(analysis)
    
    total_time = time.time() - total_start_time
    print(f"\n⚡ Total comparison time: {total_time:.1f} seconds")
    print(f"🏁 Comparison complete! Check comprehensive_comparison_results.json for details.")
    
    return all_data, analysis

if __name__ == "__main__":
    main()