#!/usr/bin/env python3
"""
Quick Test Dataset Generator
Creates small dataset for prototype testing: 1000 human + 1000 AI texts
"""

import json
import random
from pathlib import Path
import time
from typing import List, Dict

def load_human_samples(file_path: str, max_samples: int = 1000) -> List[Dict]:
    """Load random sample of human texts"""
    print(f"Loading human texts from {file_path}...")
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # Randomly sample to avoid bias
    random.shuffle(all_lines)
    
    for line in all_lines[:max_samples * 2]:  # Load extra in case of filtering
        try:
            entry = json.loads(line.strip())
            # Extract text and create standardized format
            text = entry.get('text', '')
            if text and len(text) > 100:  # Filter out very short texts
                texts.append({
                    'text': text,
                    'label': 'human',
                    'length': len(text),
                    'source': 'human_dataset'
                })
        except json.JSONDecodeError:
            continue
            
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts)} human texts")
    return texts[:max_samples]

def generate_ai_samples(human_texts: List[Dict], max_samples: int = 1000) -> List[Dict]:
    """Generate AI samples using simple transformations for quick testing"""
    print("Generating AI samples using text transformations...")
    
    ai_texts = []
    
    # Simple transformation patterns for quick testing
    patterns = [
        lambda text: text.replace("we", "the researchers").replace("our", "their").replace("We", "The researchers"),
        lambda text: text.replace("this paper", "this study").replace("This paper", "This study"),
        lambda text: text.replace("shows", "demonstrates").replace("find", "discover").replace("found", "discovered"),
        lambda text: text.replace("method", "approach").replace("technique", "methodology"),
        lambda text: text.replace("results", "findings").replace("analysis", "examination"),
        lambda text: text.replace("propose", "present").replace("novel", "new").replace("state-of-the-art", "advanced"),
        lambda text: text.replace("performance", "effectiveness").replace("accuracy", "precision"),
        lambda text: text.replace("dataset", "data collection").replace("experiment", "investigation"),
    ]
    
    for i, human_entry in enumerate(human_texts[:max_samples]):
        try:
            # Apply random transformations
            ai_text = human_entry['text']
            
            # Apply 2-4 random transformations
            selected_patterns = random.sample(patterns, random.randint(2, min(4, len(patterns))))
            for pattern in selected_patterns:
                ai_text = pattern(ai_text)
            
            # Add some variation in sentence structure
            sentences = ai_text.split('. ')
            if len(sentences) > 2:
                # Occasionally swap adjacent sentences
                if random.random() < 0.3:
                    idx = random.randint(0, len(sentences) - 2)
                    sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
                ai_text = '. '.join(sentences)
            
            # Add slight length variation
            if random.random() < 0.2:  # 20% chance to truncate slightly
                words = ai_text.split()
                truncate_at = max(len(words) - 10, int(len(words) * 0.9))
                ai_text = ' '.join(words[:truncate_at]) + '...'
            
            ai_texts.append({
                'text': ai_text,
                'label': 'ai',
                'length': len(ai_text),
                'source': 'simple_transformation',
                'original_id': i
            })
            
            if (i + 1) % 200 == 0:
                print(f"Generated {i + 1}/{max_samples} AI samples...")
                
        except Exception as e:
            print(f"Error generating AI sample {i}: {e}")
            continue
    
    print(f"Generated {len(ai_texts)} AI texts")
    return ai_texts

def create_test_dataset(input_file: str, output_file: str, samples_per_class: int = 1000):
    """Create balanced test dataset"""
    print(f"\n=== Creating Test Dataset ===")
    print(f"Target: {samples_per_class} samples per class ({samples_per_class * 2} total)")
    
    # Load human texts
    human_texts = load_human_samples(input_file, samples_per_class)
    
    # Generate AI texts
    ai_texts = generate_ai_samples(human_texts, samples_per_class)
    
    # Ensure balanced dataset
    min_samples = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:min_samples]
    ai_texts = ai_texts[:min_samples]
    
    # Combine and shuffle
    all_texts = human_texts + ai_texts
    random.shuffle(all_texts)
    
    # Save dataset
    print(f"\nSaving test dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_texts:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Statistics
    human_count = len([t for t in all_texts if t['label'] == 'human'])
    ai_count = len([t for t in all_texts if t['label'] == 'ai'])
    avg_length = sum(len(t['text']) for t in all_texts) / len(all_texts)
    
    print(f"\n=== Dataset Created ===")
    print(f"Total samples: {len(all_texts)}")
    print(f"Human samples: {human_count}")
    print(f"AI samples: {ai_count}")
    print(f"Average text length: {avg_length:.0f} characters")
    print(f"File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    print(f"Balance ratio: {human_count / ai_count:.2f}")
    
    return all_texts

def main():
    """Main execution"""
    # Configuration
    INPUT_FILE = "../human_text_50k.jsonl"  # Look in parent directory
    OUTPUT_FILE = "test_dataset_2k.jsonl"
    SAMPLES_PER_CLASS = 1000  # Small for quick testing
    
    print("🚀 Quick Test Dataset Generator")
    print("=" * 50)
    
    # Check if input file exists
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("Please ensure the human text file is in the parent directory")
        print("Expected location: ../human_text_50k.jsonl")
        return
    
    # Create dataset
    start_time = time.time()
    random.seed(42)  # For reproducible results
    dataset = create_test_dataset(INPUT_FILE, OUTPUT_FILE, SAMPLES_PER_CLASS)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Test dataset created in {elapsed:.1f} seconds")
    print(f"📁 Ready for prototype testing: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"1. Run: python ai_classifier_prototype.py")
    print(f"2. Compare SVM vs MLP vs Baseline performance")
    print(f"3. Scale up if results are promising")

if __name__ == "__main__":
    main()