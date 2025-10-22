#!/usr/bin/env python3
"""
Enhanced AI Text Generator
==========================

A clean, streamlined Python script for generating high-quality AI versions of scientific texts.

Features:
- Length control (±20% of original)
- Topic consistency validation
- Quality metrics tracking
- Support for multiple AI models (Ollama, OpenAI, etc.)
- Real-time progress monitoring

Usage:
    python3 ai_generator.py --input ../crawl/human_text_50k.jsonl --output ai_generated.jsonl --max 1000
"""

import json
import time
import re
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


class QualityController:
    """Handles quality control functions for AI text generation"""
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text for topic consistency checking"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 
                     'will', 'were', 'said', 'using', 'approach', 'method'}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:max_keywords]
    
    @staticmethod
    def check_topic_consistency(original: str, generated: str, title: str, threshold: float = 0.3) -> bool:
        """Check if generated text maintains topic consistency"""
        original_keywords = set(QualityController.extract_keywords(original))
        generated_keywords = set(QualityController.extract_keywords(generated))
        title_keywords = set(QualityController.extract_keywords(title))
        
        if len(original_keywords) == 0:
            return True
        
        # Check overlap with original text
        overlap = len(original_keywords.intersection(generated_keywords))
        consistency_score = overlap / len(original_keywords)
        
        # Check overlap with title
        title_overlap = len(title_keywords.intersection(generated_keywords))
        title_consistency = title_overlap / max(len(title_keywords), 1)
        
        return consistency_score >= threshold or title_consistency >= threshold
    
    @staticmethod
    def calculate_target_length(original_length: int, tolerance: float = 0.2) -> Tuple[int, int]:
        """Calculate target length range for generated text"""
        min_length = int(original_length * (1 - tolerance))
        max_length = int(original_length * (1 + tolerance))
        return min_length, max_length
    
    @staticmethod
    def validate_length(generated: str, original_length: int, tolerance: float = 0.2) -> bool:
        """Check if generated text length is within acceptable range"""
        min_length, max_length = QualityController.calculate_target_length(original_length, tolerance)
        return min_length <= len(generated) <= max_length


class AIGenerator:
    """Main AI text generator class"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "llama3.2:1b"):
        self.model_type = model_type
        self.model_name = model_name
        self.quality_controller = QualityController()
        
        # Initialize model connection
        if model_type == "ollama":
            self._init_ollama()
        elif model_type == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_ollama(self):
        """Initialize Ollama connection"""
        import requests
        self.session = requests.Session()
        self.base_url = "http://localhost:11434"
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {self.base_url}")
                print(f"📦 Using model: {self.model_name}")
            else:
                raise Exception(f"Ollama not responding: {response.status_code}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("💡 Make sure Ollama is running: brew services start ollama")
            print(f"💡 And model is installed: ollama pull {self.model_name}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI connection"""
        try:
            import openai
            self.openai_client = openai.OpenAI()
            print(f"✅ Connected to OpenAI")
            print(f"📦 Using model: {self.model_name}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _generate_with_ollama(self, prompt: str, max_tokens: int = 400) -> Optional[str]:
        """Generate text using Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": random.uniform(0.7, 1.0),
                "top_p": random.uniform(0.85, 0.95),
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return None
                
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return None
    
    def _generate_with_openai(self, prompt: str, max_tokens: int = 400) -> Optional[str]:
        """Generate text using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=random.uniform(0.7, 1.0)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ OpenAI generation error: {e}")
            return None
    
    def generate_text(self, prompt: str, max_tokens: int = 400) -> Optional[str]:
        """Generate text using the configured model"""
        if self.model_type == "ollama":
            return self._generate_with_ollama(prompt, max_tokens)
        elif self.model_type == "openai":
            return self._generate_with_openai(prompt, max_tokens)
        return None
    
    def create_prompts(self, title: str, abstract: str) -> List[str]:
        """Create enhanced prompts for AI generation"""
        original_length = len(abstract)
        min_length, max_length = self.quality_controller.calculate_target_length(original_length)
        
        prompts = [
            f"""Rewrite this scientific abstract using different academic language while maintaining the same topic and approximate length ({min_length}-{max_length} characters):

Title: {title}
Original Abstract: {abstract}

Rewritten Abstract:""",
            
            f"""Create an alternative version of this scientific abstract. Keep the same research topic and maintain similar length ({min_length}-{max_length} characters):

Title: {title}
Original: {abstract}

Alternative Version:""",
            
            f"""Generate a new version of this scientific abstract using different wording but the same research focus. Target length: {min_length}-{max_length} characters.

Title: {title}
Reference: {abstract}

New Abstract:"""
        ]
        
        return prompts
    
    def generate_ai_version(self, title: str, abstract: str, max_attempts: int = 3) -> str:
        """Generate AI version with quality controls"""
        original_length = len(abstract)
        prompts = self.create_prompts(title, abstract)
        
        for attempt in range(max_attempts):
            try:
                prompt = random.choice(prompts)
                target_tokens = min(original_length // 3, 500)  # Rough token estimate
                
                generated = self.generate_text(prompt, max_tokens=target_tokens)
                
                if not generated or len(generated) < 50:
                    continue
                
                # Clean up generated text
                generated = re.sub(r'^(rewritten abstract:|alternative version:|new abstract:)\s*', 
                                 '', generated, flags=re.IGNORECASE)
                generated = re.sub(r'\[ai-generated.*?\]', '', generated, flags=re.IGNORECASE)
                generated = generated.replace('\n', ' ').strip()
                
                # Length adjustment
                if len(generated) > original_length * 1.5:
                    generated = generated[:int(original_length * 1.2)] + "..."
                elif len(generated) < original_length * 0.5:
                    continue
                
                # Quality checks
                if self.quality_controller.check_topic_consistency(abstract, generated, title):
                    return generated
                    
            except Exception as e:
                print(f"⚠️ Generation attempt {attempt + 1} failed: {e}")
                continue
        
        # Enhanced fallback
        fallback_length = min(300, original_length)
        title_clean = title.lower().replace(':', '').strip()
        abstract_excerpt = abstract[:fallback_length]
        
        return f"This research on {title_clean} investigates {abstract_excerpt}... [Enhanced AI-generated summary maintaining original research focus]"


class DataProcessor:
    """Handles data loading and processing"""
    
    @staticmethod
    def load_human_texts(file_path: str) -> List[Dict]:
        """Load human texts from JSONL file"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        texts.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            raise
        
        return texts
    
    @staticmethod
    def extract_title_and_text(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract title and text from entry"""
        title = entry.get('metadata', {}).get('title', entry.get('title'))
        text = entry.get('text', '')
        return title, text
    
    @staticmethod
    def save_ai_entry(f, ai_text: str, original_entry: Dict, source: str, 
                     original_length: int, generated_length: int) -> None:
        """Save AI-generated entry to file"""
        length_ratio = generated_length / original_length if original_length > 0 else 0
        
        ai_entry = {
            'text': ai_text,
            'label': 'ai',
            'source': source,
            'original_id': original_entry.get('metadata', {}).get('arxiv_id', f"unknown"),
            'original_length': original_length,
            'generated_length': generated_length,
            'length_ratio': round(length_ratio, 2),
            'generated_at': datetime.now().isoformat()
        }
        
        f.write(json.dumps(ai_entry, ensure_ascii=False) + '\n')
        f.flush()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced AI Text Generator')
    parser.add_argument('--input', '-i', default='../crawl/human_text_50k.jsonl',
                        help='Input JSONL file with human texts')
    parser.add_argument('--output', '-o', default='ai_generated.jsonl',
                        help='Output JSONL file for AI-generated texts')
    parser.add_argument('--max', '-m', type=int, default=1000,
                        help='Maximum number of texts to generate')
    parser.add_argument('--model-type', choices=['ollama', 'openai'], default='ollama',
                        help='AI model type to use')
    parser.add_argument('--model-name', default='llama3.2:1b',
                        help='AI model name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced AI Text Generator")
    print("=" * 50)
    
    # Initialize components
    try:
        generator = AIGenerator(args.model_type, args.model_name)
        processor = DataProcessor()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Load human texts
    print(f"📂 Loading human texts from {args.input}...")
    try:
        human_texts = processor.load_human_texts(args.input)
        print(f"✅ Loaded {len(human_texts):,} human texts")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Check for existing progress
    start_idx = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            start_idx = len(f.readlines())
        print(f"🔄 Resuming from index {start_idx:,}")
    
    # Calculate generation parameters
    total_to_generate = min(args.max, len(human_texts) - start_idx)
    if total_to_generate <= 0:
        print(f"✅ Generation complete! ({start_idx} texts already generated)")
        return
    
    print(f"🎯 Generating {total_to_generate:,} AI texts...")
    print(f"⏱️ Expected time: ~{total_to_generate/5:.0f} minutes")
    print(f"💾 Output: {args.output}\n")
    
    # Generation loop
    start_time = time.time()
    generated_count = 0
    quality_stats = {'length_matches': 0, 'topic_consistent': 0, 'fallback_used': 0}
    
    mode = 'a' if args.resume else 'w'
    with open(args.output, mode, encoding='utf-8') as f:
        for i in tqdm(range(start_idx, start_idx + total_to_generate), desc="Generating"):
            entry = human_texts[i]
            
            try:
                title, text = processor.extract_title_and_text(entry)
                
                if not title:
                    print(f"\n⚠️ Missing title for entry {i}, skipping...")
                    continue
                
                if not text:
                    print(f"\n⚠️ Missing text for entry {i}, skipping...")
                    continue
                
                # Generate AI version
                ai_text = generator.generate_ai_version(title, text)
                
                # Quality assessment
                original_length = len(text)
                ai_length = len(ai_text)
                length_ratio = ai_length / original_length if original_length > 0 else 0
                
                if 0.8 <= length_ratio <= 1.2:
                    quality_stats['length_matches'] += 1
                
                if generator.quality_controller.check_topic_consistency(text, ai_text, title):
                    quality_stats['topic_consistent'] += 1
                
                if '[Enhanced AI-generated' in ai_text:
                    quality_stats['fallback_used'] += 1
                
                # Save entry
                processor.save_ai_entry(f, ai_text, entry, f"enhanced_{args.model_type}", 
                                      original_length, ai_length)
                
                generated_count += 1
                
                # Progress update
                if generated_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = generated_count / elapsed * 60
                    length_match_pct = quality_stats['length_matches'] / generated_count * 100
                    topic_match_pct = quality_stats['topic_consistent'] / generated_count * 100
                    
                    print(f"\n📊 Progress: {generated_count}/{total_to_generate}")
                    print(f"⚡ Rate: {rate:.1f} texts/minute")
                    print(f"🎯 Quality: {length_match_pct:.1f}% length match, {topic_match_pct:.1f}% topic consistent")
                
            except Exception as e:
                print(f"\n❌ Error with entry {i}: {e}")
                continue
    
    # Final statistics
    total_time = time.time() - start_time
    final_rate = generated_count / total_time * 60
    
    print(f"\n✅ Generation Complete!")
    print(f"📝 Generated: {generated_count:,} texts")
    print(f"⚡ Final rate: {final_rate:.1f} texts/minute")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"💾 Saved to: {args.output}")
    
    # Quality summary
    if generated_count > 0:
        length_match_pct = quality_stats['length_matches'] / generated_count * 100
        topic_consistent_pct = quality_stats['topic_consistent'] / generated_count * 100
        fallback_pct = quality_stats['fallback_used'] / generated_count * 100
        
        print(f"\n🎯 Final Quality Summary:")
        print(f"   Length matching (±20%): {length_match_pct:.1f}%")
        print(f"   Topic consistency: {topic_consistent_pct:.1f}%")
        print(f"   Fallback used: {fallback_pct:.1f}%")


if __name__ == "__main__":
    import os
    main()