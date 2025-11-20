#!/usr/bin/env python3
"""
API Key Testing Script
Tests all configured API providers to verify connectivity and authentication
"""

import os
import yaml
import requests
import json
from typing import Dict, Any
from pathlib import Path


def load_env_file(env_path: str = ".env"):
    """Load environment variables from .env file"""
    # Convert string or Path to Path object
    if isinstance(env_path, str):
        env_file = Path(env_path)
    else:
        env_file = env_path
    
    if not env_file.exists():
        print(f"⚠ Warning: .env file not found at {env_file.absolute()}")
        return False
    
    print(f"✓ Loading environment variables from {env_file.absolute()}\n")
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set environment variable
                    os.environ[key] = value
                    print(f"  ✓ Loaded: {key}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error loading .env file: {e}\n")
        return False


class APITester:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the API tester with configuration"""
        self.config_path = config_path
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {self.config_path}\n")
            return config
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            return {}
    
    def get_api_key(self, provider: str) -> str:
        """Get API key from environment variables"""
        env_var_map = {
            'groq': 'GROQ_API_KEY',
            'sambanova': 'SAMBANOVA_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'firework': 'FIREWORK_API_KEY'  # Changed from FIREWORKS_API_KEY to match .env
        }
        
        env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")
        api_key = os.getenv(env_var)
        
        if not api_key:
            print(f"  ⚠ API key not found in environment: {env_var}")
            return None
        
        # Mask the API key for display
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"  ✓ API key found: {masked_key}")
        return api_key
    
    def test_groq(self) -> Dict[str, Any]:
        """Test Groq API"""
        print("\n" + "="*60)
        print("Testing GROQ API")
        print("="*60)
        
        api_key = self.get_api_key('groq')
        if not api_key:
            return {"status": "failed", "error": "API key not found"}
        
        config = self.config.get('api_providers', {}).get('groq', {})
        base_url = config.get('base_url', 'https://api.groq.com/openai/v1')
        model = config.get('models', {}).get('primary', 'llama-3.1-8b-instant')
        
        try:
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, respond with just 'Hi'"}],
                "max_tokens": 10
            }
            
            print(f"  → Making request to: {url}")
            print(f"  → Using model: {model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"  → Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"  ✓ SUCCESS! Response: {content}")
                return {"status": "success", "response": content, "model": model}
            else:
                error_msg = response.text
                print(f"  ✗ FAILED! Error: {error_msg}")
                return {"status": "failed", "error": error_msg, "status_code": response.status_code}
                
        except Exception as e:
            print(f"  ✗ EXCEPTION: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def test_sambanova(self) -> Dict[str, Any]:
        """Test SambaNova API"""
        print("\n" + "="*60)
        print("Testing SAMBANOVA API")
        print("="*60)
        
        api_key = self.get_api_key('sambanova')
        if not api_key:
            return {"status": "failed", "error": "API key not found"}
        
        config = self.config.get('api_providers', {}).get('sambanova', {})
        base_url = config.get('base_url', 'https://api.sambanova.ai/v1')
        model = config.get('models', {}).get('primary', 'Meta-Llama-3.1-8B-Instruct')
        
        try:
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, respond with just 'Hi'"}],
                "max_tokens": 10
            }
            
            print(f"  → Making request to: {url}")
            print(f"  → Using model: {model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"  → Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"  ✓ SUCCESS! Response: {content}")
                return {"status": "success", "response": content, "model": model}
            else:
                error_msg = response.text
                print(f"  ✗ FAILED! Error: {error_msg}")
                return {"status": "failed", "error": error_msg, "status_code": response.status_code}
                
        except Exception as e:
            print(f"  ✗ EXCEPTION: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def test_cohere(self) -> Dict[str, Any]:
        """Test Cohere API"""
        print("\n" + "="*60)
        print("Testing COHERE API")
        print("="*60)
        
        api_key = self.get_api_key('cohere')
        if not api_key:
            return {"status": "failed", "error": "API key not found"}
        
        config = self.config.get('api_providers', {}).get('cohere', {})
        base_url = config.get('base_url', 'https://api.cohere.ai/v1')
        model = config.get('models', {}).get('primary', 'command')
        
        try:
            url = f"{base_url}/generate"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "prompt": "Hello, respond with just 'Hi'",
                "max_tokens": 10
            }
            
            print(f"  → Making request to: {url}")
            print(f"  → Using model: {model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"  → Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('generations', [{}])[0].get('text', '')
                print(f"  ✓ SUCCESS! Response: {content}")
                return {"status": "success", "response": content, "model": model}
            else:
                error_msg = response.text
                print(f"  ✗ FAILED! Error: {error_msg}")
                return {"status": "failed", "error": error_msg, "status_code": response.status_code}
                
        except Exception as e:
            print(f"  ✗ EXCEPTION: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def test_openrouter(self) -> Dict[str, Any]:
        """Test OpenRouter API"""
        print("\n" + "="*60)
        print("Testing OPENROUTER API")
        print("="*60)
        
        api_key = self.get_api_key('openrouter')
        if not api_key:
            return {"status": "failed", "error": "API key not found"}
        
        config = self.config.get('api_providers', {}).get('openrouter', {})
        base_url = config.get('base_url', 'https://openrouter.ai/api/v1')
        model = config.get('models', {}).get('primary', 'meta-llama/llama-3.1-8b-instruct:free')
        
        # Try recommended model if the configured one fails
        fallback_models = [
            model,
            "google/gemini-flash-1.5",
            "mistralai/mistral-7b-instruct:free"
        ]
        
        for test_model in fallback_models:
            try:
                url = f"{base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",  # Required by OpenRouter
                    "X-Title": "AI Detector Test"
                }
                payload = {
                    "model": test_model,
                    "messages": [{"role": "user", "content": "Hello, respond with just 'Hi'"}],
                    "max_tokens": 10
                }
                
                print(f"  → Making request to: {url}")
                print(f"  → Using model: {test_model}")
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                print(f"  → Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(f"  ✓ SUCCESS! Response: {content}")
                    if test_model != model:
                        print(f"  ℹ Suggestion: Update config.yaml to use '{test_model}' instead of '{model}'")
                    return {"status": "success", "response": content, "model": test_model}
                else:
                    error_msg = response.text
                    print(f"  ✗ FAILED with {test_model}! Error: {error_msg}")
                    if test_model != fallback_models[-1]:
                        print(f"  → Trying next fallback model...")
                    continue
                    
            except Exception as e:
                print(f"  ✗ EXCEPTION with {test_model}: {str(e)}")
                if test_model != fallback_models[-1]:
                    print(f"  → Trying next fallback model...")
                continue
        
        # If all models failed
        return {"status": "failed", "error": "All models failed", "models_tested": fallback_models}
    
    def test_firework(self) -> Dict[str, Any]:
        """Test Fireworks API"""
        print("\n" + "="*60)
        print("Testing FIREWORKS API")
        print("="*60)
        
        api_key = self.get_api_key('firework')
        if not api_key:
            return {"status": "failed", "error": "API key not found"}
        
        config = self.config.get('api_providers', {}).get('firework', {})
        base_url = config.get('base_url', 'https://api.fireworks.ai/inference/v1')
        model = config.get('models', {}).get('primary', 'accounts/fireworks/models/llama-v3p1-8b-instruct')
        
        try:
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, respond with just 'Hi'"}],
                "max_tokens": 10
            }
            
            print(f"  → Making request to: {url}")
            print(f"  → Using model: {model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"  → Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"  ✓ SUCCESS! Response: {content}")
                return {"status": "success", "response": content, "model": model}
            else:
                error_msg = response.text
                print(f"  ✗ FAILED! Error: {error_msg}")
                return {"status": "failed", "error": error_msg, "status_code": response.status_code}
                
        except Exception as e:
            print(f"  ✗ EXCEPTION: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def run_all_tests(self):
        """Run tests for all configured API providers"""
        print("\n" + "="*60)
        print("API KEY TESTING SCRIPT")
        print("="*60)
        
        providers = self.config.get('api_providers', {})
        
        # Test each provider
        test_methods = {
            'groq': self.test_groq,
            'sambanova': self.test_sambanova,
            'cohere': self.test_cohere,
            'openrouter': self.test_openrouter,
            'firework': self.test_firework
        }
        
        for provider, config in providers.items():
            enabled = config.get('enabled', False)
            
            if provider in test_methods:
                if enabled:
                    self.results[provider] = test_methods[provider]()
                else:
                    print(f"\n{'='*60}")
                    print(f"Skipping {provider.upper()} (disabled in config)")
                    print("="*60)
                    self.results[provider] = {"status": "skipped", "reason": "disabled in config"}
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for provider, result in self.results.items():
            status = result.get('status', 'unknown')
            
            if status == 'success':
                print(f"  ✓ {provider.upper()}: SUCCESS")
                success_count += 1
            elif status == 'skipped':
                print(f"  ⊘ {provider.upper()}: SKIPPED ({result.get('reason', 'unknown')})")
                skipped_count += 1
            else:
                print(f"  ✗ {provider.upper()}: FAILED")
                error = result.get('error', 'Unknown error')
                print(f"      Error: {error[:100]}...")
                failed_count += 1
        
        print("\n" + "-"*60)
        print(f"Total: {len(self.results)} providers")
        print(f"Success: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Skipped: {skipped_count}")
        print("="*60)


def main():
    """Main function to run the API tests"""
    # Try to find and load .env file from current directory or project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Try current directory first, then project root
    if not load_env_file(current_dir / ".env"):
        load_env_file(project_root / ".env")
    
    tester = APITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
