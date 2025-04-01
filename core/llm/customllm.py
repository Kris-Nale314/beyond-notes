# core/llm/customllm.py
import json
import asyncio
import logging
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

# Define a type alias for the usage dictionary for clarity
UsageDict = Dict[str, Optional[int]]

class CustomLLM:
    """
    LLM interface using OpenAI SDK 1.x, modified to return token usage.
    Handles basic async requests and structured output generation via function calling.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM interface.

        Args:
            api_key: API key for the LLM provider (OpenAI).
            model: Model name to use (default: gpt-3.5-turbo).
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.model = model
        logger.info(f"CustomLLM initialized with model: {self.model}")

        # Initialize sync client (useful for simple scripts or tests)
        try:
            self._sync_client = OpenAI(api_key=api_key)
        except Exception as e:
             logger.error(f"Failed to initialize synchronous OpenAI client: {e}")
             # Decide if this should be fatal or just prevent sync usage
             self._sync_client = None

        # Async client initialized lazily
        self._async_client = None

    def _get_async_client(self) -> AsyncOpenAI:
        """Lazy initializes and returns the async OpenAI client."""
        if self._async_client is None:
            try:
                self._async_client = AsyncOpenAI(api_key=self.api_key)
                logger.info("AsyncOpenAI client initialized.")
            except Exception as e:
                 logger.error(f"Failed to initialize asynchronous OpenAI client: {e}")
                 raise RuntimeError("Could not initialize AsyncOpenAI client") from e
        return self._async_client

    def _parse_usage(self, response: ChatCompletion) -> UsageDict:
        """Safely parse the usage dictionary from an OpenAI response."""
        usage_data = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        if response and hasattr(response, 'usage') and response.usage:
            usage = response.usage
            usage_data["prompt_tokens"] = getattr(usage, 'prompt_tokens', None)
            usage_data["completion_tokens"] = getattr(usage, 'completion_tokens', None)
            usage_data["total_tokens"] = getattr(usage, 'total_tokens', None)
            logger.debug(f"Parsed token usage: {usage_data}")
        else:
            logger.warning("Could not find usage information in LLM response.")
        return usage_data

    async def _make_llm_call_with_retry(self, **kwargs) -> ChatCompletion:
        """Internal helper to make the API call with a simple retry on rate limit."""
        client = self._get_async_client()
        try:
            response = await client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            # Check specifically for OpenAI's RateLimitError if possible, otherwise string match
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit encountered for model {self.model}. Retrying after 3 seconds...")
                await asyncio.sleep(3)
                try:
                    response = await client.chat.completions.create(**kwargs)
                    logger.info("Retry successful after rate limit.")
                    return response
                except Exception as retry_e:
                     logger.error(f"LLM call failed even after retry: {retry_e}")
                     raise retry_e # Re-raise the exception after retry failure
            else:
                logger.error(f"LLM call failed: {e}")
                raise e # Re-raise other exceptions immediately

    async def generate_completion(self,
                                 prompt: str,
                                 system_prompt: Optional[str] = None,
                                 max_tokens: int = 1500,
                                 temperature: float = 0.5) -> Tuple[str, UsageDict]:
        """
        Generate a text completion from the LLM.

        Args:
            prompt: The main user prompt.
            system_prompt: Optional system message to guide the assistant's behavior.
            max_tokens: Maximum number of tokens for the completion.
            temperature: Sampling temperature.

        Returns:
            A tuple containing:
            - The generated text completion (str).
            - A dictionary with token usage information (UsageDict).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Requesting completion from {self.model} with max_tokens={max_tokens}, temp={temperature}")
        response = await self._make_llm_call_with_retry(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        result_str = ""
        if response.choices:
            result_str = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        else:
            logger.warning("LLM response did not contain any choices.")

        usage_dict = self._parse_usage(response)
        return (result_str, usage_dict)

    def _sanitize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize the schema to ensure it's compatible with OpenAI's function calling API.
        
        This handles common issues with JSON Schema validation in the OpenAI API.
        
        Args:
            schema: The original schema dictionary
            
        Returns:
            A sanitized version of the schema
        """
        # Create a deep copy to avoid modifying the original
        sanitized = json.loads(json.dumps(schema))
        
        # Make root type "object" if not already specified
        if "type" not in sanitized:
            sanitized["type"] = "object"
            
        # Ensure properties is a dictionary if type is object
        if sanitized.get("type") == "object" and "properties" not in sanitized:
            sanitized["properties"] = {}
            
        # Handle schema with only array/items at the root
        if "items" in sanitized and "type" not in sanitized["items"]:
            sanitized["items"]["type"] = "object"
            
        # Check all properties and their types recursively
        self._sanitize_schema_properties(sanitized)
            
        return sanitized
    
    def _sanitize_schema_properties(self, schema_obj: Dict[str, Any]) -> None:
        """
        Recursively sanitize all properties in the schema.
        
        Args:
            schema_obj: Schema or sub-schema object to sanitize in-place
        """
        # Handle schema properties
        if "properties" in schema_obj and isinstance(schema_obj["properties"], dict):
            for prop_name, prop_schema in schema_obj["properties"].items():
                # Ensure each property has a type
                if "type" not in prop_schema:
                    # Default to string for untyped properties
                    prop_schema["type"] = "string"
                
                # Recursively sanitize nested objects
                if prop_schema.get("type") == "object":
                    self._sanitize_schema_properties(prop_schema)
                    
                # Handle array items
                if prop_schema.get("type") == "array" and "items" in prop_schema:
                    # Ensure items has a type
                    if isinstance(prop_schema["items"], dict) and "type" not in prop_schema["items"]:
                        prop_schema["items"]["type"] = "object"
                    # Recursively sanitize array item schema
                    if isinstance(prop_schema["items"], dict):
                        self._sanitize_schema_properties(prop_schema["items"])
        
        # Handle items for array types
        if "items" in schema_obj and isinstance(schema_obj["items"], dict):
            # Ensure items has a type
            if "type" not in schema_obj["items"]:
                schema_obj["items"]["type"] = "object"
            # Process items object recursively
            self._sanitize_schema_properties(schema_obj["items"])

    async def generate_structured_output(self,
                                        prompt: str,
                                        output_schema: Dict[str, Any],
                                        system_prompt: Optional[str] = None,
                                        max_tokens: int = 2000,
                                        temperature: float = 0.2) -> Tuple[Dict[str, Any], UsageDict]:
        """
        Generate structured output (JSON) from the LLM using function calling.

        Args:
            prompt: The main user prompt.
            output_schema: JSON schema defining the desired output structure.
            system_prompt: Optional system message.
            max_tokens: Maximum tokens for the completion (includes function call).
            temperature: Sampling temperature (often lower for structured output).

        Returns:
            A tuple containing:
            - The generated dictionary matching the schema (Dict[str, Any]).
            - A dictionary with token usage information (UsageDict).
        Raises:
            ValueError: If the LLM fails to return valid function call arguments or parsable JSON.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_name = "extract_structured_data"
        
        # Sanitize schema to ensure it's compatible with OpenAI's API
        sanitized_schema = self._sanitize_schema(output_schema)

        logger.debug(f"Requesting structured output from {self.model} matching schema. Max_tokens={max_tokens}, temp={temperature}")
        try:
            response = await self._make_llm_call_with_retry(
                model=self.model,
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Extract structured information based on the provided schema.",
                        "parameters": sanitized_schema
                    }
                }],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            # Log the schema if there's an error
            logger.error(f"Error with schema: {json.dumps(sanitized_schema, indent=2)}")
            raise e

        usage_dict = self._parse_usage(response)
        result_dict = {}

        message = response.choices[0].message if response.choices else None
        if message and message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name == tool_name:
                try:
                    arguments = tool_call.function.arguments
                    result_dict = json.loads(arguments)
                    logger.info(f"Successfully parsed structured output from function call '{tool_name}'.")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON arguments from function call: {arguments}. Error: {e}")
                    raise ValueError(f"LLM returned invalid JSON in function arguments: {e}") from e
            else:
                logger.error(f"LLM called unexpected function: {tool_call.function.name}")
                raise ValueError(f"LLM called unexpected function: {tool_call.function.name}")
        else:
            # Fallback for when function calling fails
            logger.warning(f"LLM did not return the expected function call ('{tool_name}'). Attempting to parse content as JSON.")
            content = message.content.strip() if message and message.content else "{}"
            try:
                result_dict = json.loads(content)
                logger.info("Parsed LLM content as JSON fallback.")
            except json.JSONDecodeError:
                logger.error(f"LLM fallback content was not valid JSON: {content}")
                raise ValueError("LLM did not use the function call and content was not valid JSON.")

        return (result_dict, usage_dict)

    # --- Sync method (optional update) ---
    def generate_completion_sync(self,
                               prompt: str,
                               max_tokens: int = 1000,
                               temperature: float = 0.7) -> Tuple[str, UsageDict]:
        """Synchronous version for basic testing, also returning usage."""
        if not self._sync_client:
             logger.error("Synchronous client not initialized.")
             return ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        try:
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            result_str = response.choices[0].message.content.strip() if response.choices else ""
            usage_dict = self._parse_usage(response)
            return (result_str, usage_dict)
        except Exception as e:
            logger.error(f"Sync LLM call failed: {e}")
            return ("", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})