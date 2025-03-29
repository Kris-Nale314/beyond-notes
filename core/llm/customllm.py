# core/llm/customllm.py
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Optional, List, Union
import os

class CustomLLM:
    """
    Simple LLM interface for the application.
    Compatible with OpenAI SDK 1.x
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM interface.
        
        Args:
            api_key: API key for the LLM provider
            model: Model to use (default: gpt-3.5-turbo)
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize sync client with minimal parameters
        self._sync_client = OpenAI(api_key=api_key)
        
        # Async client will be initialized when needed
        self._async_client = None
    
    def _get_async_client(self):
        """Get the async client, initializing if needed."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client
    
    async def generate_completion(self, 
                                 prompt: str, 
                                 max_tokens: int = 1000,
                                 temperature: float = 0.7) -> str:
        """
        Generate a text completion from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Simple retry for rate limits
            if "rate limit" in str(e).lower():
                await asyncio.sleep(2)
                client = self._get_async_client()
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            else:
                raise
    
    async def generate_structured_output(self,
                                        prompt: str,
                                        json_schema: Dict[str, Any],
                                        max_tokens: int = 1000,
                                        temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a structured output in JSON format from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            json_schema: JSON schema defining the expected response structure
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated JSON response
        """
        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "extract_information",
                        "description": "Extract structured information from the text",
                        "parameters": json_schema
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "extract_information"}},
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract function call results from the response
            message = response.choices[0].message
            if message.tool_calls and len(message.tool_calls) > 0:
                # Parse the function arguments
                function_args = message.tool_calls[0].function.arguments
                return json.loads(function_args)
            else:
                # Fallback if no tool/function call was made
                text_response = message.content.strip()
                try:
                    return json.loads(text_response)
                except:
                    return {"text": text_response}
            
        except Exception as e:
            raise

    def generate_completion_sync(self, 
                               prompt: str, 
                               max_tokens: int = 1000,
                               temperature: float = 0.7) -> str:
        """Synchronous version of generate_completion for simple testing."""
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()