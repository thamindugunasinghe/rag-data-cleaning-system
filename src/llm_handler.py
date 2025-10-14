import openai
from typing import List, Dict, Optional
import yaml
import os
from dotenv import load_dotenv
import time

load_dotenv()

class LLMHandler:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables!")
        else:
            print(f"✅ OpenAI API key loaded (ends with: ...{api_key[-4:]})")
            
        self.client = openai.OpenAI(api_key=api_key)
        
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Config loaded: Model={self.config['llm']['model']}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            # Default config
            self.config = {
                'llm': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.1,
                    'max_tokens': 100
                }
            }
    
    def predict_missing_value(self, row_context: Dict, column_name: str, data_type: str) -> str:
        """Use LLM to predict missing values based on context"""
        
        print(f"    🤖 Calling OpenAI API for '{column_name}' prediction...")
        
        if not row_context:
            print(f"    ⚠️  No context available for prediction")
            return None
        
        if data_type == 'quantitative':
            prompt = self._create_quantitative_prompt(row_context, column_name)
        else:
            prompt = self._create_qualitative_prompt(row_context, column_name)
        
        print(f"    📝 Prompt length: {len(prompt)} characters")
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are a data cleaning expert. Predict missing values based on context. Return only the predicted value, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens'],
                timeout=30  # 30 second timeout
            )
            
            api_time = time.time() - start_time
            prediction = response.choices[0].message.content.strip()
            
            print(f"    ⏱️  API call took {api_time:.2f} seconds")
            print(f"    🎯 Raw prediction: '{prediction}'")
            
            # Clean up prediction
            prediction = prediction.replace('"', '').replace("'", '').strip()
            
            return prediction
        
        except openai.APITimeoutError:
            print(f"    ⏰ API timeout (30s) for column '{column_name}'")
            return None
            
        except openai.APIError as e:
            print(f"    ❌ OpenAI API error: {e}")
            return None
            
        except Exception as e:
            print(f"    ❌ Unexpected error during LLM prediction: {e}")
            return None
    
    def _create_quantitative_prompt(self, context: Dict, column_name: str) -> str:
        """Create prompt for quantitative data prediction"""
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if v is not None])
        
        return f"""
        Given the following data context: {context_str}
        
        Predict the missing value for '{column_name}' (numeric value).
        Consider the relationships between the variables and provide a realistic estimate.
        Return only the number, no explanation.
        """
    
    def _create_qualitative_prompt(self, context: Dict, column_name: str) -> str:
        """Create prompt for qualitative data prediction"""
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if v is not None])
        
        return f"""
        Given the following data context: {context_str}
        
        Predict the missing value for '{column_name}' (categorical value).
        Consider the patterns and relationships in the data.
        Return only the predicted category, no explanation.
        """
    
    def validate_business_rules(self, row: Dict, rules: List[str]) -> Dict:
        """Use LLM to validate business rules"""
        
        print(f"🔍 Validating business rules for row...")
        
        rules_str = "\n".join([f"- {rule}" for rule in rules])
        row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
        
        prompt = f"""
        Validate this data row against business rules:
        
        Data: {row_str}
        
        Rules to check:
        {rules_str}
        
        Return a JSON object with:
        {{"valid": true/false, "violations": ["rule1", "rule2"], "confidence": 0.0-1.0}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=[
                    {"role": "system", "content": "You are a data validation expert. Check data against business rules and return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=15
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            print(f"✅ Validation completed: {result}")
            return result
        
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return {"valid": True, "violations": [], "confidence": 0.5}
