#!/usr/bin/env python3
"""
Data Preprocessing Pipeline
==========================

This script combines human and AI text datasets, cleans the data, and creates 
train/validation/test splits ready for machine learning training.

Usage:
    python3 preprocessing.py
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_datasets(self, human_file="../data/crawl/human_text_50k.jsonl", ai_file="../data/generate/ai_text_50k_colab.jsonl"):
        """Load and combine human and AI datasets"""
        
        print("Loading datasets...")
        
        # Load human texts
        human_texts = []
        try:
            with open(human_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    human_texts.append({
                        'text': entry.get('text', ''),
                        'label': 'human',
                        'source': entry.get('source', 'arxiv'),
                        'id': entry.get('id', '')
                    })
            print(f"Loaded {len(human_texts):,} human texts")
        except FileNotFoundError:
            print(f"Human file not found: {human_file}")
            return False
            
        # Load AI texts
        ai_texts = []
        try:
            with open(ai_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    ai_texts.append({
                        'text': entry.get('text', ''),
                        'label': 'ai', 
                        'source': entry.get('source', 'generated'),
                        'id': entry.get('original_id', '')
                    })
            print(f"Loaded {len(ai_texts):,} AI texts")
        except FileNotFoundError:
            print(f"AI file not found: {ai_file}")
            print("Generate AI texts first using the Colab notebook!")
            return False
        
        # Combine datasets
        all_texts = human_texts + ai_texts
        self.data = pd.DataFrame(all_texts)
        
        print(f"Combined dataset: {len(self.data):,} total texts")
        print(f"   Human: {len(human_texts):,} ({len(human_texts)/len(all_texts)*100:.1f}%)")
        print(f"   AI: {len(ai_texts):,} ({len(ai_texts)/len(all_texts)*100:.1f}%)")
        
        return True
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short or very long texts
        if len(text) < 50 or len(text) > 5000:
            return ""
            
        # Basic cleaning (keep scientific formatting)
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_data(self):
        """Clean and preprocess the combined dataset"""
        
        print("\nCleaning and preprocessing data...")
        
        original_count = len(self.data)
        
        # Clean text
        self.data['clean_text'] = self.data['text'].apply(self.clean_text)
        
        # Remove empty texts
        self.data = self.data[self.data['clean_text'].str.len() > 0]
        
        # Remove duplicates
        before_dedup = len(self.data)
        self.data = self.data.drop_duplicates(subset=['clean_text'], keep='first')
        duplicates_removed = before_dedup - len(self.data)
        
        # Calculate text statistics
        self.data['text_length'] = self.data['clean_text'].str.len()
        self.data['word_count'] = self.data['clean_text'].str.split().str.len()
        
        print(f"Preprocessing results:")
        print(f"   Original texts: {original_count:,}")
        print(f"   After cleaning: {len(self.data):,}")
        print(f"   Duplicates removed: {duplicates_removed:,}")
        print(f"   Final dataset: {len(self.data):,} texts")
        
        # Show statistics by label
        stats = self.data.groupby('label').agg({
            'text_length': ['mean', 'std'],
            'word_count': ['mean', 'std']
        }).round(2)
        
        print(f"\nText Statistics by Label:")
        print(stats)
        
        return True
    
    def create_splits(self, test_size=0.2, val_size=0.2):
        """Create train/validation/test splits"""
        
        print(f"\nCreating data splits...")
        print(f"   Test size: {test_size*100:.0f}%")
        print(f"   Validation size: {val_size*100:.0f}%") 
        print(f"   Train size: {(1-test_size-val_size)*100:.0f}%")
        
        # First split: train+val vs test
        train_val_data, test_data = train_test_split(
            self.data,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.data['label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)  # Adjust for remaining data
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val_data['label']
        )
        
        self.train_data = train_data.reset_index(drop=True)
        self.val_data = val_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)
        
        print(f"Data splits created:")
        print(f"   Train: {len(self.train_data):,} texts")
        print(f"   Validation: {len(self.val_data):,} texts") 
        print(f"   Test: {len(self.test_data):,} texts")
        
        # Verify label distribution
        for name, data in [("Train", self.train_data), ("Val", self.val_data), ("Test", self.test_data)]:
            label_counts = data['label'].value_counts()
            print(f"   {name}: {label_counts['human']} human, {label_counts['ai']} AI")
        
        return True
    
    def save_processed_data(self, output_dir="../processed_data"):
        """Save processed datasets"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}/...")
        
        # Save as JSON Lines (for text data)
        datasets = {
            'train': self.train_data,
            'val': self.val_data, 
            'test': self.test_data,
            'full': self.data
        }
        
        for name, data in datasets.items():
            # Save as JSONL
            output_file = f"{output_dir}/{name}_data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for _, row in data.iterrows():
                    json.dump({
                        'text': row['clean_text'],
                        'label': row['label'],
                        'source': row['source'],
                        'text_length': int(row['text_length']),
                        'word_count': int(row['word_count'])
                    }, f)
                    f.write('\n')
            print(f"   {name}_data.jsonl: {len(data):,} texts")
            
            # Save as CSV (for easy viewing)
            csv_file = f"{output_dir}/{name}_data.csv"
            data[['clean_text', 'label', 'source', 'text_length', 'word_count']].to_csv(
                csv_file, index=False
            )
        
        # Save preprocessing stats
        stats_file = f"{output_dir}/preprocessing_stats.json"
        stats = {
            'preprocessing_date': datetime.now().isoformat(),
            'total_texts': len(self.data),
            'train_size': len(self.train_data),
            'val_size': len(self.val_data),
            'test_size': len(self.test_data),
            'human_texts': int(self.data[self.data['label'] == 'human'].shape[0]),
            'ai_texts': int(self.data[self.data['label'] == 'ai'].shape[0]),
            'avg_text_length': float(self.data['text_length'].mean()),
            'avg_word_count': float(self.data['word_count'].mean())
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"   preprocessing_stats.json")
        print(f"\nAll processed data saved to {output_dir}/")
        
        return True
    
    def create_visualizations(self, output_dir="processed_data"):
        """Create data visualization plots"""
        
        print(f"\nCreating data visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Text length distribution
            axes[0,0].hist(self.data[self.data['label']=='human']['text_length'], 
                          alpha=0.7, label='Human', bins=50, color='blue')
            axes[0,0].hist(self.data[self.data['label']=='ai']['text_length'], 
                          alpha=0.7, label='AI', bins=50, color='red')
            axes[0,0].set_xlabel('Text Length (characters)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title('Text Length Distribution')
            axes[0,0].legend()
            
            # Word count distribution  
            axes[0,1].hist(self.data[self.data['label']=='human']['word_count'],
                          alpha=0.7, label='Human', bins=50, color='blue')
            axes[0,1].hist(self.data[self.data['label']=='ai']['word_count'],
                          alpha=0.7, label='AI', bins=50, color='red')
            axes[0,1].set_xlabel('Word Count')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Word Count Distribution')
            axes[0,1].legend()
            
            # Label distribution
            label_counts = self.data['label'].value_counts()
            axes[1,0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
                         colors=['blue', 'red'])
            axes[1,0].set_title('Label Distribution')
            
            # Dataset splits
            split_data = {
                'Train': len(self.train_data),
                'Validation': len(self.val_data), 
                'Test': len(self.test_data)
            }
            axes[1,1].bar(split_data.keys(), split_data.values(), color=['green', 'orange', 'purple'])
            axes[1,1].set_ylabel('Number of Texts')
            axes[1,1].set_title('Dataset Split Sizes')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/data_analysis.png", dpi=300, bbox_inches='tight')
            print(f"   data_analysis.png saved")
            
            plt.close()
            
        except Exception as e:
            print(f"   Could not create plots: {e}")
            print("   Install matplotlib and seaborn: pip install matplotlib seaborn")

def main():
    """Main preprocessing pipeline"""
    
    print("Human vs AI Text Classification - Data Preprocessing")
    print("="*55)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load datasets
    if not preprocessor.load_datasets():
        return
    
    # Preprocess data
    if not preprocessor.preprocess_data():
        return
    
    # Create train/val/test splits
    if not preprocessor.create_splits():
        return
    
    # Save processed data
    if not preprocessor.save_processed_data():
        return
    
    # Create visualizations
    preprocessor.create_visualizations()
    
    print("\nPreprocessing pipeline completed successfully!")
    print("\nNext steps:")
    print("   1. Review processed_data/ directory")
    print("   2. Run training pipeline: python3 train_models.py")
    print("   3. Compare model performance")

if __name__ == "__main__":
    main()