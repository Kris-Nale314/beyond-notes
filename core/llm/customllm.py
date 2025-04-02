# core/llm/customllm.py
import json
import asyncio
import logging
import re # Import regex for parsing
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

# Define a type alias for the usage dictionary for clarity
UsageDict = Dict[str, Optional[int]]

class CustomLLM:
    """
    LLM interface using OpenAI SDK 1.x.
    Handles async/sync requests, structured output via tool calling,
    and an alternative structured output method via prompt formatting and parsing.
    Includes default temperature setting and basic retry logic.
    """

    DEFAULT_TEMP = 0.5 # Default temperature if not specified

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-3.5-turbo",
                 default_temperature: float = DEFAULT_TEMP):
        """
        Initialize the LLM interface.

        Args:
            api_key: API key for the LLM provider (OpenAI).
            model: Model name to use (default: gpt-3.5-turbo).
            default_temperature: Default sampling temperature for requests.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        logger.info(f"CustomLLM initialized with model: {self.model}, default_temp: {self.default_temperature}")

        # Initialize sync client (useful for simple scripts or tests)
        try:
            self._sync_client = OpenAI(api_key=api_key)
        except Exception as e:
             logger.error(f"Failed to initialize synchronous OpenAI client: {e}", exc_info=True)
             self._sync_client = None # Allow operation without sync client, log error

        # Async client initialized lazily
        self._async_client = None

    def _get_async_client(self) -> AsyncOpenAI:
        """Lazy initializes and returns the async OpenAI client."""
        if self._async_client is None:
            try:
                self._async_client = AsyncOpenAI(api_key=self.api_key)
                logger.info("AsyncOpenAI client initialized.")
            except Exception as e:
                 logger.error(f"Failed to initialize asynchronous OpenAI client: {e}", exc_info=True)
                 raise RuntimeError("Could not initialize AsyncOpenAI client") from e
        return self._async_client

    def _parse_usage(self, response: ChatCompletion) -> UsageDict:
        """Safely parse the usage dictionary from an OpenAI response."""
        # (Implementation remains the same as before)
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

    async def _make_llm_call_with_retry(self, is_async: bool = True, **kwargs) -> ChatCompletion:
        """Internal helper to make the API call with retry, supports sync/async."""
        max_retries = 1
        retry_delay_seconds = 3

        for attempt in range(max_retries + 1):
            try:
                if is_async:
                    client = self._get_async_client()
                    response = await client.chat.completions.create(**kwargs)
                else:
                    if not self._sync_client:
                        raise RuntimeError("Synchronous client not initialized.")
                    client = self._sync_client
                    response = client.chat.completions.create(**kwargs)
                # If successful, return response
                if attempt > 0:
                     logger.info(f"Retry successful after rate limit (attempt {attempt + 1}).")
                return response
            except Exception as e:
                # Check specifically for OpenAI's RateLimitError if possible, otherwise string match
                is_rate_limit = "rate limit" in str(e).lower() # Basic check
                # Add more specific error checks if needed (e.g., from openai.error import RateLimitError)

                if is_rate_limit and attempt < max_retries:
                    logger.warning(f"Rate limit encountered for model {self.model} (attempt {attempt + 1}). Retrying after {retry_delay_seconds} seconds...")
                    if is_async:
                        await asyncio.sleep(retry_delay_seconds)
                    else:
                        time.sleep(retry_delay_seconds) # Use time.sleep for sync
                else:
                    # Log and re-raise if not a rate limit error or retries exhausted
                    logger.error(f"LLM call failed (attempt {attempt + 1}): {e}", exc_info=(not is_rate_limit)) # Show traceback for non-rate limit errors
                    raise e
        # This line should theoretically not be reached if exceptions are raised correctly
        raise RuntimeError("LLM call failed after maximum retries.")


    # --- Standard Completion ---

    async def generate_completion(self,
                                 prompt: str,
                                 system_prompt: Optional[str] = None,
                                 max_tokens: int = 1500,
                                 temperature: Optional[float] = None # Made optional
                                 ) -> Tuple[str, UsageDict]:
        """
        Generate a text completion from the LLM (async).

        Args:
            prompt: The main user prompt.
            system_prompt: Optional system message to guide the assistant's behavior.
            max_tokens: Maximum number of tokens for the completion.
            temperature: Sampling temperature (uses default if None).

        Returns:
            Tuple[Generated text string, Usage dictionary]
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.default_temperature

        logger.debug(f"Requesting async completion from {self.model} with max_tokens={max_tokens}, temp={temp}")
        response = await self._make_llm_call_with_retry(
            is_async=True,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp
        )

        result_str = ""
        if response.choices:
            result_str = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        else:
            logger.warning("LLM response did not contain any choices.")

        usage_dict = self._parse_usage(response)
        return (result_str, usage_dict)

    def generate_completion_sync(self,
                                 prompt: str,
                                 system_prompt: Optional[str] = None, # Added system prompt
                                 max_tokens: int = 1000,
                                 temperature: Optional[float] = None # Made optional
                                 ) -> Tuple[str, UsageDict]:
        """Synchronous version for text completion."""
        if not self._sync_client:
             logger.error("Synchronous client not initialized.")
             return ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.default_temperature

        logger.debug(f"Requesting sync completion from {self.model} with max_tokens={max_tokens}, temp={temp}")
        try:
            # Use sync retry helper
            response = self._make_llm_call_with_retry(
                is_async=False,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp
            )
            result_str = response.choices[0].message.content.strip() if response.choices else ""
            usage_dict = self._parse_usage(response)
            return (result_str, usage_dict)
        except Exception as e:
            # Error already logged in _make_llm_call_with_retry
            # Return empty/none result
            return ("", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})


    # --- Structured Output via Tool/Function Calling ---

    def _sanitize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize the schema for OpenAI function calling compatibility."""
        # (Implementation remains the same as before)
        sanitized = json.loads(json.dumps(schema))
        if "type" not in sanitized: sanitized["type"] = "object"
        if sanitized.get("type") == "object" and "properties" not in sanitized: sanitized["properties"] = {}
        if "items" in sanitized and isinstance(sanitized.get("items"), dict) and "type" not in sanitized["items"]: sanitized["items"]["type"] = "object" # Basic array item check
        self._sanitize_schema_properties(sanitized)
        return sanitized

    def _sanitize_schema_properties(self, schema_obj: Dict[str, Any]) -> None:
        """Recursively sanitize schema properties."""
         # (Implementation remains the same as before)
        if "properties" in schema_obj and isinstance(schema_obj["properties"], dict):
            for prop_name, prop_schema in schema_obj["properties"].items():
                if isinstance(prop_schema, dict): # Ensure prop_schema is a dict before accessing keys
                     if "type" not in prop_schema: prop_schema["type"] = "string" # Default type
                     if prop_schema.get("type") == "object": self._sanitize_schema_properties(prop_schema)
                     if prop_schema.get("type") == "array" and "items" in prop_schema and isinstance(prop_schema["items"], dict):
                         if "type" not in prop_schema["items"]: prop_schema["items"]["type"] = "object" # Default item type
                         self._sanitize_schema_properties(prop_schema["items"])
        if "items" in schema_obj and isinstance(schema_obj["items"], dict):
            if "type" not in schema_obj["items"]: schema_obj["items"]["type"] = "object"
            self._sanitize_schema_properties(schema_obj["items"])

    async def generate_structured_output(self,
                                        prompt: str,
                                        output_schema: Dict[str, Any],
                                        system_prompt: Optional[str] = None,
                                        max_tokens: int = 2000,
                                        temperature: Optional[float] = None # Made optional
                                        ) -> Tuple[Dict[str, Any], UsageDict]:
        """
        Generate structured output (JSON) using Tool/Function calling (async).

        Args:
            prompt: The main user prompt.
            output_schema: JSON schema defining the desired output structure.
            system_prompt: Optional system message.
            max_tokens: Maximum tokens for the completion.
            temperature: Sampling temperature (uses default if None).

        Returns:
            Tuple[Generated dictionary matching schema, Usage dictionary]
        Raises:
            ValueError: If LLM fails to return valid function call arguments or parsable JSON.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_name = "extract_structured_data"
        sanitized_schema = self._sanitize_schema(output_schema)
        temp = temperature if temperature is not None else self.default_temperature

        logger.debug(f"Requesting async structured output from {self.model} matching schema. Max_tokens={max_tokens}, temp={temp}")
        try:
            response = await self._make_llm_call_with_retry(
                is_async=True,
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
                tool_choice={"type": "function", "function": {"name": tool_name}}, # Force the function call
                max_tokens=max_tokens,
                temperature=temp
            )
        except Exception as e:
            logger.error(f"Error during LLM call for structured output. Schema: {json.dumps(sanitized_schema, indent=2)}")
            raise e # Re-raise after logging schema

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
                # This case should be less likely with tool_choice="required" or specific func name
                logger.error(f"LLM called unexpected function: {tool_call.function.name}")
                raise ValueError(f"LLM called unexpected function: {tool_call.function.name}")
        else:
             # Fallback logic from previous version (attempt to parse raw content)
            logger.warning(f"LLM did not return the expected function call ('{tool_name}'). Attempting to parse raw content as JSON.")
            content = message.content.strip() if message and message.content else ""
            if content:
                 # Try extracting JSON from potential markdown code blocks first
                 parsed_fallback = self._extract_and_parse_json(content)
                 if parsed_fallback is not None:
                      result_dict = parsed_fallback
                      logger.info("Parsed LLM content fallback as JSON (extracted from potential markdown).")
                 else:
                      logger.error(f"LLM fallback content could not be parsed as JSON: {content[:200]}...")
                      raise ValueError("LLM did not use function call and fallback content was not valid/extractable JSON.")
            else:
                 logger.error("LLM did not use function call and returned no fallback content.")
                 raise ValueError("LLM did not use the function call and content was empty.")


        return (result_dict, usage_dict)


    # --- Alternative Structured Output via Prompting & Parsing ---

    def _extract_and_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Tries to extract a JSON string potentially embedded in text (e.g., markdown)
        and parse it.
        """
        # Regex to find JSON within ```json ... ``` blocks, stripping the markers
        match = re.search(r"```(?:json)?\s*({.*?}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: If no markdown block, assume the whole text might be JSON (or attempt to find start/end braces)
            # More sophisticated extraction might be needed for robustness
             start_brace = text.find('{')
             start_bracket = text.find('[')
             if start_brace == -1 and start_bracket == -1: return None # No JSON start found

             if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                  # Assume object starts at first '{' and find corresponding '}'
                  json_str = text[start_brace:]
                  # Basic balancing - might fail for complex strings within JSON
                  # Consider using a more robust JSON parser that handles partial strings if needed
             elif start_bracket != -1:
                  # Assume array starts at first '['
                  json_str = text[start_bracket:]
             else:
                  return None # Should not happen

             # Attempt to find matching end bracket/brace (basic implementation)
             # This is fragile; a dedicated JSON parsing library is better for complex cases
             # json_str = json_str # Basic approach: hope json.loads handles extra trailing chars (it sometimes does)

        if not json_str:
            return None

        try:
            # Sanitize common issues before parsing (like escaped newlines if needed)
            # json_str = json_str.replace('\\n', '\n')
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extracted JSON string: {e}. String (partial): {json_str[:200]}...")
            return None

    async def generate_and_parse_json(self,
                                      prompt: str,
                                      description: str = "the required data", # Description of what the JSON should contain
                                      system_prompt: Optional[str] = None,
                                      max_tokens: int = 1500,
                                      temperature: Optional[float] = None # Made optional
                                      ) -> Tuple[Optional[Dict[str, Any]], UsageDict]:
        """
        Generates text completion and attempts to parse JSON from the response (async).
        Uses prompting to ask the LLM for JSON output, often within markdown markers.

        Args:
            prompt: The main user prompt, describing the task.
            description: A short description of the JSON structure needed (for the prompt).
            system_prompt: Optional system message.
            max_tokens: Maximum tokens for the completion.
            temperature: Sampling temperature (uses default if None).

        Returns:
             Tuple[Parsed dictionary or None if parsing fails, Usage dictionary]
        """
        temp = temperature if temperature is not None else self.default_temperature

        # Modify the user prompt to explicitly ask for JSON output
        json_instruction = f"""\nPlease provide the response strictly as a valid JSON object containing {description}.
Enclose the JSON object within triple backticks, like this:
```json
{{
  "key": "value",
  ...
}}
```
Respond ONLY with the JSON object within the backticks."""

        full_prompt = f"{prompt}\n{json_instruction}"

        logger.debug(f"Requesting async completion (expecting JSON) from {self.model}. Max_tokens={max_tokens}, temp={temp}")

        # Call the standard text completion method
        raw_text, usage_dict = await self.generate_completion(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temp
        )

        if not raw_text:
            logger.warning("LLM returned empty content when asked for JSON.")
            return (None, usage_dict)

        # Attempt to extract and parse JSON from the raw text
        parsed_json = self._extract_and_parse_json(raw_text)

        if parsed_json is None:
            logger.warning(f"Failed to extract or parse JSON from LLM response. Raw text (partial): {raw_text[:200]}...")

        return (parsed_json, usage_dict)


    def generate_and_parse_json_sync(self,
                                      prompt: str,
                                      description: str = "the required data",
                                      system_prompt: Optional[str] = None,
                                      max_tokens: int = 1500,
                                      temperature: Optional[float] = None
                                      ) -> Tuple[Optional[Dict[str, Any]], UsageDict]:
        """Synchronous version of generate_and_parse_json."""
        if not self._sync_client:
             logger.error("Synchronous client not initialized.")
             return (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        temp = temperature if temperature is not None else self.default_temperature

        json_instruction = f"""\nPlease provide the response strictly as a valid JSON object containing {description}.
Enclose the JSON object within triple backticks, like this:
```json
{{ ... }}
```
Respond ONLY with the JSON object within the backticks."""
        full_prompt = f"{prompt}\n{json_instruction}"

        logger.debug(f"Requesting sync completion (expecting JSON) from {self.model}. Max_tokens={max_tokens}, temp={temp}")

        raw_text, usage_dict = self.generate_completion_sync(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temp
        )

        if not raw_text:
             logger.warning("LLM returned empty content (sync) when asked for JSON.")
             return (None, usage_dict)

        parsed_json = self._extract_and_parse_json(raw_text)
        if parsed_json is None:
            logger.warning(f"Failed to extract or parse JSON from sync LLM response. Raw text (partial): {raw_text[:200]}...")

        return (parsed_json, usage_dict)

