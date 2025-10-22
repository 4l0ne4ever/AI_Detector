#!/usr/bin/env python3
"""
Multi-API AI Text Generator
==========================

A robust system for generating AI texts using multiple free LLM APIs with auto-failover,
quality control, and progress tracking.

Usage:
    python multi_api_generator.py --input ../crawl/human_text_50k.jsonl --output ai_generated_50k.jsonl --max 50000
"""

import json
import re
import os
import time
import random
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import yaml
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try to load .env manually
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of text generation attempt"""
    text: str
    success: bool
    provider: str
    model: str
    temperature: float
    attempt: int
    error: Optional[str] = None


class APIManager:
    """Manages multiple LLM APIs with auto-failover and rate limiting"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.providers = self._initialize_providers()
        self.rate_limits = {provider: {'requests': 0, 'last_reset': time.time()} 
                          for provider in self.providers.keys()}
        
        # Provider balancing tracking
        self.balancing_config = self.config.get('provider_balancing', {})
        self.balancing_enabled = self.balancing_config.get('enabled', False)
        self.target_distribution = self.balancing_config.get('target_distribution', {})
        self.tolerance = self.balancing_config.get('tolerance', 0.05)
        self.rebalance_threshold = self.balancing_config.get('rebalance_threshold', 0.10)
        
        # Usage tracking for balancing
        self.provider_usage = {provider: 0 for provider in self.providers.keys()}
        self.total_requests = 0
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def _initialize_providers(self) -> Dict:
        """Initialize API providers based on configuration"""
        providers = {}
        
        for provider_name, config in self.config['api_providers'].items():
            if not config.get('enabled', False):
                continue
                
            api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
            if not api_key:
                logger.warning(f"No API key found for {provider_name}, skipping")
                continue
                
            providers[provider_name] = {
                'config': config,
                'api_key': api_key,
                'base_url': config['base_url'],
                'models': config['models'],
                'rate_limit': config['rate_limit']
            }
            
        if not providers:
            raise ValueError("No valid API providers configured")
            
        logger.info(f"Initialized {len(providers)} API providers: {list(providers.keys())}")
        return providers
    
    def _check_rate_limit(self, provider: str) -> bool:
        """Check if provider is within rate limits"""
        now = time.time()
        rate_config = self.providers[provider]['rate_limit']
        rate_info = self.rate_limits[provider]
        
        # Reset counter if minute has passed
        if now - rate_info['last_reset'] >= 60:
            rate_info['requests'] = 0
            rate_info['last_reset'] = now
        
        # Check per-minute limit
        if rate_info['requests'] >= rate_config['requests_per_minute']:
            return False
            
        return True
    
    def _increment_rate_limit(self, provider: str):
        """Increment request counter for provider"""
        self.rate_limits[provider]['requests'] += 1
    
    def _get_available_provider(self) -> Optional[str]:
        """Get next available provider based on balancing, priority and rate limits"""
        if not self.balancing_enabled or self.total_requests < 10:
            # Use priority-based selection for first few requests
            return self._get_provider_by_priority()
        
        # Use balancing-based selection
        return self._get_provider_by_balancing()
    
    def _get_provider_by_priority(self) -> Optional[str]:
        """Get provider based on priority (original logic)"""
        sorted_providers = sorted(
            self.providers.items(),
            key=lambda x: x[1]['config']['priority']
        )
        
        for provider_name, provider_info in sorted_providers:
            if self._check_rate_limit(provider_name):
                return provider_name
                
        return None
    
    def _get_provider_by_balancing(self) -> Optional[str]:
        """Get provider based on usage balancing"""
        if self.total_requests == 0:
            return self._get_provider_by_priority()
        
        # Calculate current distribution
        current_distribution = {}
        for provider in self.providers.keys():
            current_distribution[provider] = self.provider_usage[provider] / self.total_requests
        
        # Find providers that are under their target quota
        available_providers = []
        for provider_name in self.providers.keys():
            if not self._check_rate_limit(provider_name):
                continue
                
            current_ratio = current_distribution.get(provider_name, 0)
            target_ratio = self.target_distribution.get(provider_name, 0.25)
            
            # Check if provider is under quota (with tolerance)
            if current_ratio < target_ratio + self.tolerance:
                available_providers.append((provider_name, target_ratio - current_ratio))
        
        if not available_providers:
            # Fallback to priority if no balanced providers available
            return self._get_provider_by_priority()
        
        # Sort by how much under quota they are (most under quota first)
        available_providers.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most under-quota provider
        return available_providers[0][0]
    
    def _update_provider_usage(self, provider_name: str):
        """Update usage statistics for balancing"""
        self.provider_usage[provider_name] += 1
        self.total_requests += 1
        
        # Log balancing info every 100 requests
        if self.total_requests % 100 == 0:
            self._log_balancing_stats()
    
    def _log_balancing_stats(self):
        """Log current balancing statistics"""
        if self.total_requests == 0:
            return
            
        logger.info(f"Provider Balancing Stats (after {self.total_requests} requests):")
        for provider in self.providers.keys():
            current_ratio = self.provider_usage[provider] / self.total_requests
            target_ratio = self.target_distribution.get(provider, 0.25)
            deviation = abs(current_ratio - target_ratio)
            
            status = "✅" if deviation <= self.tolerance else "⚠️" if deviation <= self.rebalance_threshold else "❌"
            logger.info(f"  {provider}: {current_ratio:.1%} (target: {target_ratio:.1%}) {status}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def _call_groq_api(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call Groq API"""
        provider = self.providers['groq']
        headers = {
            'Authorization': f'Bearer {provider["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': random.uniform(0.85, 0.95)
        }
        
        response = requests.post(
            f"{provider['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=provider['config']['timeout']
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            raise requests.RequestException(f"Groq API error: {response.status_code} - {response.text}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def _call_huggingface_api(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call Hugging Face Inference API"""
        provider = self.providers['huggingface']
        headers = {
            'Authorization': f'Bearer {provider["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'inputs': prompt,
            'parameters': {
                'max_length': max_tokens,
                'temperature': temperature,
                'do_sample': True,
                'top_p': random.uniform(0.85, 0.95)
            }
        }
        
        response = requests.post(
            f"{provider['base_url']}/{model}",
            headers=headers,
            json=data,
            timeout=provider['config']['timeout']
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            elif isinstance(result, dict):
                return result.get('generated_text', '').strip()
            return str(result).strip()
        else:
            raise requests.RequestException(f"Hugging Face API error: {response.status_code} - {response.text}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def _call_cohere_api(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call Cohere API"""
        provider = self.providers['cohere']
        headers = {
            'Authorization': f'Bearer {provider["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'p': random.uniform(0.85, 0.95)
        }
        
        response = requests.post(
            f"{provider['base_url']}/generate",
            headers=headers,
            json=data,
            timeout=provider['config']['timeout']
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('generations', [{}])[0].get('text', '').strip()
        else:
            raise requests.RequestException(f"Cohere API error: {response.status_code} - {response.text}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def _call_openrouter_api(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call OpenRouter API"""
        provider = self.providers['openrouter']
        headers = {
            'Authorization': f'Bearer {provider["api_key"]}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/your-repo',  # Optional
            'X-Title': 'AI Text Generator'  # Optional
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': random.uniform(0.85, 0.95)
        }
        
        response = requests.post(
            f"{provider['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=provider['config']['timeout']
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            raise requests.RequestException(f"OpenRouter API error: {response.status_code} - {response.text}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def _call_firework_api(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call Firework AI API"""
        provider = self.providers['firework']
        headers = {
            'Authorization': f'Bearer {provider["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': random.uniform(0.85, 0.95)
        }
        
        response = requests.post(
            f"{provider['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=provider['config']['timeout']
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            raise requests.RequestException(f"Firework AI API error: {response.status_code} - {response.text}")
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> GenerationResult:
        """Generate text using available provider with auto-failover"""
        temperature = random.uniform(*self.config['generation_settings']['temperature_range'])
        
        for attempt in range(self.config['generation_settings']['max_attempts']):
            provider_name = self._get_available_provider()
            
            if not provider_name:
                logger.warning("No available providers, waiting...")
                time.sleep(60)  # Wait 1 minute before retry
                continue
            
            try:
                provider = self.providers[provider_name]
                model = provider['models']['primary']
                
                # Call appropriate API method
                if provider_name == 'groq':
                    text = self._call_groq_api(prompt, model, temperature, max_tokens)
                elif provider_name == 'huggingface':
                    text = self._call_huggingface_api(prompt, model, temperature, max_tokens)
                elif provider_name == 'cohere':
                    text = self._call_cohere_api(prompt, model, temperature, max_tokens)
                elif provider_name == 'openrouter':
                    text = self._call_openrouter_api(prompt, model, temperature, max_tokens)
                elif provider_name == 'firework':
                    text = self._call_firework_api(prompt, model, temperature, max_tokens)
                else:
                    raise ValueError(f"Unknown provider: {provider_name}")
                
                # Increment rate limit counter
                self._increment_rate_limit(provider_name)
                
                # Update usage for balancing
                self._update_provider_usage(provider_name)
                
                return GenerationResult(
                    text=text,
                    success=True,
                    provider=provider_name,
                    model=model,
                    temperature=temperature,
                    attempt=attempt + 1
                )
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed (attempt {attempt + 1}): {e}")
                continue
        
        # All attempts failed
        return GenerationResult(
            text="",
            success=False,
            provider="none",
            model="none",
            temperature=temperature,
            attempt=self.config['generation_settings']['max_attempts'],
            error="All providers failed"
        )


class PromptBuilder:
    """Builds diverse prompts for AI text generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.templates = config['prompt_templates']
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text with improved scientific term handling"""
        import re
        
        # Extract words (including hyphenated terms and numbers)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]\b|\b[a-zA-Z]{3,}\b', text.lower())
        
        word_freq = {}
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'were', 'said',
            'using', 'approach', 'method', 'study', 'research', 'paper', 'work', 'results',
            'analysis', 'data', 'based', 'show', 'propose', 'present', 'demonstrate',
            'investigate', 'examine', 'explore', 'develop', 'design', 'implement',
            'evaluate', 'compare', 'consider', 'provide', 'include', 'contain',
            'different', 'various', 'several', 'multiple', 'numerous', 'various',
            'important', 'significant', 'effective', 'efficient', 'successful',
            'novel', 'new', 'recent', 'current', 'existing', 'previous', 'future'
        }
        
        for word in words:
            if word not in stop_words and len(word) >= 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:max_keywords]
    
    def calculate_target_length(self, original_length: int) -> Tuple[int, int]:
        """Calculate target length range"""
        tolerance = self.config['quality_thresholds']['length_tolerance']
        min_length = int(original_length * (1 - tolerance))
        max_length = int(original_length * (1 + tolerance))
        return min_length, max_length
    
    def build_prompt(self, title: str, text: str, categories: List[str], template_name: str = None) -> str:
        """Build prompt using specified template"""
        if template_name is None:
            template_name = random.choice(list(self.templates.keys()))
        
        template = self.templates[template_name]
        original_length = len(text)
        min_length, max_length = self.calculate_target_length(original_length)
        
        # Format categories for display
        categories_str = ', '.join(categories) if categories else 'General'
        
        # Extract keywords for keyword-based templates
        keywords = self.extract_keywords(text, max_keywords=8)
        keywords_str = ', '.join(keywords) if keywords else 'machine learning, artificial intelligence'
        
        # Calculate target length in words (approximate)
        target_length_words = max_length // 5  # Rough estimate: 5 chars per word
        
        # Build prompt based on template type
        if template_name in ['template_d', 'template_e', 'template_f', 'template_g']:
            # Templates that don't use original text
            prompt = template['format'].format(
                title=title,
                categories=categories_str,
                keywords=keywords_str,
                target_length=target_length_words
            )
        else:
            # Templates that use original text
            prompt = template['format'].format(
                title=title,
                text=text,
                categories=categories_str,
                keywords=keywords_str,
                min_length=min_length,
                max_length=max_length,
                target_length=target_length_words
            )
        
        return prompt, template_name, min_length, max_length


class QualityValidator:
    """Validates generated text quality"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config['quality_thresholds']
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text"""
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 
                     'will', 'were', 'said', 'using', 'approach', 'method', 'study',
                     'research', 'paper', 'work', 'results', 'analysis', 'data'}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:max_keywords]
    
    def validate_length(self, generated_text: str, original_length: int) -> bool:
        """Validate generated text length"""
        generated_length = len(generated_text)
        tolerance = self.thresholds['length_tolerance']
        
        min_length = int(original_length * (1 - tolerance))
        max_length = int(original_length * (1 + tolerance))
        
        return (min_length <= generated_length <= max_length and 
                generated_length >= self.thresholds['min_generated_length'] and
                generated_length <= self.thresholds['max_generated_length'])
    
    def validate_topic_consistency(self, original_text: str, generated_text: str, title: str) -> bool:
        """Validate topic consistency between original and generated text"""
        original_keywords = set(self.extract_keywords(original_text))
        generated_keywords = set(self.extract_keywords(generated_text))
        title_keywords = set(self.extract_keywords(title))
        
        if len(original_keywords) == 0:
            return True
        
        # Check overlap with original text
        overlap = len(original_keywords.intersection(generated_keywords))
        consistency_score = overlap / len(original_keywords)
        
        # Check overlap with title
        title_overlap = len(title_keywords.intersection(generated_keywords))
        title_consistency = title_overlap / max(len(title_keywords), 1)
        
        threshold = self.thresholds['topic_consistency_threshold']
        return consistency_score >= threshold or title_consistency >= threshold
    
    def detect_prompt_leakage(self, text: str) -> bool:
        """Detect prompt leakage patterns"""
        leakage_patterns = [
            r'rewritten abstract:',
            r'alternative version:',
            r'new abstract:',
            r'here is',
            r'below is',
            r'as an ai',
            r'i cannot',
            r'\[ai-generated',
            r'\(generated by',
            r'note: this is',
            r'---',
            r'###',
            r'##',
            r'# abstract:',
            r'# rewritten:',
            r'abstract:',
            r'summary:',
            r'version:',
            r'alternative:',
            r'here\'s',
            r'here is the',
            r'this is a',
            r'this abstract',
            r'the following',
            r'below you will find'
        ]
        
        text_lower = text.lower()
        for pattern in leakage_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def detect_repetition(self, text: str) -> bool:
        """Detect excessive repetition"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return False
        
        # Check for repeated sentences
        unique_sentences = set(sentences)
        repetition_ratio = len(unique_sentences) / len(sentences)
        
        if repetition_ratio < 0.8:  # More than 20% repetition
            return True
        
        # Check for repeated n-grams
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = {}
        for bigram in bigrams:
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
        
        max_bigram_count = max(bigram_counts.values()) if bigram_counts else 0
        if max_bigram_count > len(words) * 0.1:  # Same bigram appears >10% of text
            return True
        
        return False
    
    def check_scientific_style(self, text: str) -> bool:
        """Check if text maintains scientific writing style"""
        text_lower = text.lower()
        
        # Check for excessive first-person
        first_person_count = len(re.findall(r'\b(i|we|my|our|me|us)\b', text_lower))
        word_count = len(text.split())
        if word_count > 0 and first_person_count / word_count > 0.05:  # >5% first person
            return False
        
        # Check for informal language
        informal_words = ['awesome', 'cool', 'amazing', 'great', 'nice', 'good', 'bad', 
                         'terrible', 'wow', 'yeah', 'ok', 'okay', 'sure', 'definitely',
                         'totally', 'really', 'very', 'so', 'quite', 'pretty']
        
        informal_count = sum(1 for word in informal_words if word in text_lower)
        if informal_count > 3:  # More than 3 informal words
            return False
        
        # Check for conversational tone
        conversational_patterns = [
            r'you can',
            r'you will',
            r'you should',
            r'let me',
            r'let\'s',
            r'i think',
            r'i believe',
            r'in my opinion',
            r'as you can see',
            r'as we know'
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def validate_quality(self, original_text: str, generated_text: str, title: str) -> Dict[str, bool]:
        """Comprehensive quality validation"""
        original_length = len(original_text)
        
        return {
            'length_valid': self.validate_length(generated_text, original_length),
            'topic_consistent': self.validate_topic_consistency(original_text, generated_text, title),
            'not_empty': len(generated_text.strip()) > 0,
            'not_too_short': len(generated_text) >= self.thresholds['min_generated_length'],
            'no_prompt_leakage': not self.detect_prompt_leakage(generated_text),
            'no_repetition': not self.detect_repetition(generated_text),
            'scientific_style': self.check_scientific_style(generated_text)
        }


class AIGenerator:
    """Main AI text generator orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.api_manager = APIManager(config_path)
        self.prompt_builder = PromptBuilder(self.api_manager.config)
        self.quality_validator = QualityValidator(self.api_manager.config)
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'provider_usage': {provider: 0 for provider in self.api_manager.providers.keys()},
            'resumed_from_checkpoint': 0
        }
        
        # Daily limits
        self.daily_limit = 15000  # Max samples per day
        self.current_day_samples = 0
        
        # Checkpointing
        self.checkpoint_file = None
        self.processed_ids = set()
    
    def _load_checkpoint(self, checkpoint_file: str) -> set:
        """Load processed IDs from checkpoint file"""
        processed_ids = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    for line in f:
                        processed_ids.add(line.strip())
                logger.info(f"Loaded checkpoint: {len(processed_ids)} already processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return processed_ids
    
    def _save_checkpoint(self, entry_id: str):
        """Save processed ID to checkpoint file"""
        if self.checkpoint_file:
            try:
                with open(self.checkpoint_file, 'a') as f:
                    f.write(f"{entry_id}\n")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
    
    def _check_daily_limit(self) -> bool:
        """Check if daily limit reached"""
        return self.current_day_samples >= self.daily_limit
    
    def _reset_daily_counter(self):
        """Reset daily counter for new day"""
        self.current_day_samples = 0
    
    def process_single_text(self, entry: Dict) -> Optional[Dict]:
        """Process a single human text entry"""
        try:
            # Extract data from entry
            text = entry.get('text', '')
            metadata = entry.get('metadata', {})
            title = metadata.get('title', 'Untitled')
            categories = metadata.get('categories', [])
            arxiv_id = metadata.get('arxiv_id', 'unknown')
            
            if not text or len(text.strip()) < 50:
                logger.warning(f"Skipping entry with insufficient text: {arxiv_id}")
                return None
            
            # Build prompt
            prompt, template_name, min_length, max_length = self.prompt_builder.build_prompt(
                title, text, categories
            )
            
            # Generate AI text
            max_tokens = min(int(len(text) * self.api_manager.config['generation_settings']['max_tokens_multiplier']), 1000)
            result = self.api_manager.generate_text(prompt, max_tokens)
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['provider_usage'][result.provider] += 1
            
            if not result.success:
                self.stats['failed_generations'] += 1
                logger.error(f"Generation failed for {arxiv_id}: {result.error}")
                return None
            
            # Clean generated text
            generated_text = self._clean_generated_text(result.text)
            
            # Quality validation
            quality_checks = self.quality_validator.validate_quality(text, generated_text, title)
            
            # Check if quality is acceptable based on config
            quality_config = self.api_manager.config.get('quality_control', {})
            strict_mode = quality_config.get('strict_mode', False)
            required_checks = quality_config.get('required_checks', ['length_valid', 'topic_consistent', 'not_empty'])
            
            if strict_mode:
                # Strict mode: ALL required checks must pass
                quality_passed = all(quality_checks.get(check, True) for check in required_checks)
            else:
                # Lenient mode: Most checks should pass
                quality_passed = sum(quality_checks.values()) >= len(quality_checks) * 0.8
            
            if not quality_passed:
                self.stats['quality_failed'] += 1
                logger.warning(f"Quality validation failed for {arxiv_id}: {quality_checks}")
                return None  # Reject this sample
            else:
                self.stats['quality_passed'] += 1
            
            # Create output entry
            ai_entry = {
                'text': generated_text,
                'label': 'ai',
                'source_llm': f"{result.provider}-{result.model}",
                'temperature': result.temperature,
                'prompt_template': template_name,
                'original_id': arxiv_id,
                'original_title': title,
                'original_categories': categories,
                'original_length': len(text),
                'generated_length': len(generated_text),
                'length_ratio': round(len(generated_text) / len(text), 2) if len(text) > 0 else 0,
                'generation_attempt': result.attempt,
                'quality_checks': quality_checks,
                'generated_at': datetime.now().isoformat()
            }
            
            self.stats['successful_generations'] += 1
            return ai_entry
            
        except Exception as e:
            logger.error(f"Error processing entry {entry.get('metadata', {}).get('arxiv_id', 'unknown')}: {e}")
            self.stats['failed_generations'] += 1
            return None
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text with comprehensive pattern removal"""
        import re
        
        # Remove common prefixes and markers
        patterns_to_remove = [
            r'^(rewritten abstract:|alternative version:|new abstract:|here is|below is|here\'s|this is a|this abstract|the following|below you will find)\s*',
            r'\[ai-generated.*?\]',
            r'\(generated by.*?\)',
            r'note: this is.*?$',
            r'---+\s*$',
            r'###+\s*$',
            r'##+\s*$',
            r'# abstract:',
            r'# rewritten:',
            r'abstract:',
            r'summary:',
            r'version:',
            r'alternative:',
            r'as an ai.*?$',
            r'i cannot.*?$',
            r'let me.*?$',
            r'let\'s.*?$'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove quotes if the entire text is wrapped in quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        return text.strip()
    
    def generate_batch(self, input_file: str, output_file: str, max_samples: int = None, 
                     start_index: int = 0, resume: bool = False) -> Dict:
        """Generate AI texts for a batch of human texts with daily limits and checkpointing"""
        
        # Setup checkpointing
        self.checkpoint_file = f"{output_file}.checkpoint"
        self.processed_ids = self._load_checkpoint(self.checkpoint_file)
        self.stats['resumed_from_checkpoint'] = len(self.processed_ids)
        
        # Load human texts
        logger.info(f"Loading human texts from {input_file}")
        human_texts = self._load_human_texts(input_file)
        
        # Filter out already processed texts
        texts_to_process = []
        for entry in human_texts:
            entry_id = entry.get('metadata', {}).get('arxiv_id', '')
            if entry_id not in self.processed_ids:
                texts_to_process.append(entry)
        
        logger.info(f"Processing {len(texts_to_process)} texts (skipping {len(self.processed_ids)} already processed)")
        
        if max_samples:
            texts_to_process = texts_to_process[:max_samples]
        
        total_to_process = len(texts_to_process)
        
        # Check daily limit
        if self._check_daily_limit():
            logger.warning("Daily limit reached! Stopping generation.")
            return {'successful': 0, 'total': 0, 'rate': 0, 'time': 0}
        
        # Process texts
        successful_count = 0
        start_time = time.time()
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for i, entry in enumerate(tqdm(texts_to_process, desc="Generating AI texts")):
                # Check daily limit before each generation
                if self._check_daily_limit():
                    logger.warning(f"Daily limit reached after {successful_count} generations. Stopping.")
                    break
                
                ai_entry = self.process_single_text(entry)
                
                if ai_entry:
                    f.write(json.dumps(ai_entry, ensure_ascii=False) + '\n')
                    f.flush()
                    successful_count += 1
                    self.stats['successful_generations'] += 1
                    self.current_day_samples += 1  # Track daily samples
                    
                    # Save checkpoint
                    entry_id = entry.get('metadata', {}).get('arxiv_id', '')
                    self._save_checkpoint(entry_id)
                else:
                    self.stats['failed_generations'] += 1
                
                # Progress reporting
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60
                    success_rate = successful_count / (i + 1) * 100
                    
                    logger.info(f"Progress: {i + 1}/{total_to_process} "
                              f"({rate:.1f} texts/min, {success_rate:.1f}% success)")
        
        # Final statistics
        total_time = time.time() - start_time
        final_rate = successful_count / total_time * 60 if total_time > 0 else 0
        
        logger.info(f"Generation complete!")
        logger.info(f"Successfully generated: {successful_count}/{total_to_process}")
        logger.info(f"Final rate: {final_rate:.1f} texts/minute")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        
        return {
            'successful': successful_count,
            'total': total_to_process,
            'rate': final_rate,
            'time': total_time / 60,
            'stats': self.stats
        }
    
    def _load_human_texts(self, file_path: str) -> List[Dict]:
        """Load human texts from JSONL file"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        texts.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        
        return texts


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-API AI Text Generator')
    parser.add_argument('--input', '-i', required=True,
                        help='Input JSONL file with human texts')
    parser.add_argument('--output', '-o', required=True,
                        help='Output JSONL file for AI-generated texts')
    parser.add_argument('--max', '-m', type=int, default=15000,
                        help='Maximum number of texts to generate per day (default: 15000)')
    parser.add_argument('--day', type=int, default=1,
                        help='Current day (1-4)')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start processing from this index')
    
    args = parser.parse_args()
    
    print("🚀 Multi-API AI Text Generator - Daily System")
    print("=" * 50)
    print(f"📅 Day {args.day} Generation")
    print(f"🎯 Target: {args.max} samples")
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print()
    
    logger.info("Multi-API AI Text Generator")
    logger.info("=" * 50)
    
    try:
        generator = AIGenerator(args.config)
        
        # Check daily limit
        if generator._check_daily_limit():
            print("❌ Daily limit already reached!")
            return
        
        result = generator.generate_batch(
            args.input,
            args.output,
            args.max,
            args.start_index,
            args.resume
        )

        # Report results
        print(f"\n✅ Day {args.day} Complete!")
        print(f"📈 Generated: {result['successful']} samples")
        print(f"⏱️  Rate: {result['rate']:.1f} texts/minute")
        print(f"🕐 Time: {result['time']:.1f} minutes")
        
        # Check if we should continue tomorrow
        if result['successful'] >= args.max:
            print(f"\n🎯 Daily target reached! Run again tomorrow with --day {args.day + 1}")
        else:
            print(f"\n⚠️  Daily target not reached. You can run again with same --day {args.day}")
        
        logger.info("Generation completed successfully!")
        logger.info("Final Statistics:")
        logger.info(f"   Successful: {result['successful']}/{result['total']}")
        logger.info(f"   Rate: {result['rate']:.1f} texts/minute")
        logger.info(f"   Provider usage: {generator.stats['provider_usage']}")
        
        print(f"\n🎉 Anti-bias features active:")
        print(f"   ✅ Provider balancing enabled")
        print(f"   ✅ Diverse prompt templates")
        print(f"   ✅ Comprehensive quality validation")
        print(f"   ✅ Checkpointing and resume support")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
