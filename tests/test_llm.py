# standalone_test.py
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("ERROR: No OpenAI API key found in environment variables.")
    exit(1)

print(f"API key found: {api_key[:5]}...{api_key[-4:]}")
print(f"OpenAI API Base URL: {os.environ.get('OPENAI_API_BASE', 'default')}")
print(f"Proxy settings: {os.environ.get('HTTP_PROXY', 'none')}, {os.environ.get('HTTPS_PROXY', 'none')}")

# Test with direct OpenAI client
print("\n--- Testing OpenAI API directly ---")
try:
    from openai import OpenAI
    
    # Explicitly create client with minimal parameters
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print("Direct API test successful!")
except Exception as e:
    print(f"ERROR in direct API test: {e}")

# Print installed packages
print("\n--- OpenAI Package Information ---")
try:
    import openai
    print(f"OpenAI version: {openai.__version__}")
    print(f"OpenAI module path: {openai.__path__}")
    
    import pkg_resources
    openai_pkg = pkg_resources.get_distribution("openai")
    print(f"OpenAI package distribution: {openai_pkg}")
    
    httpx_pkg = pkg_resources.get_distribution("httpx")
    print(f"HTTPX package distribution: {httpx_pkg}")
except Exception as e:
    print(f"Error getting package info: {e}")

# Also test async for completeness
print("\n--- Testing Async OpenAI API ---")

async def test_async_openai():
    try:
        from openai import AsyncOpenAI
        
        # Create async client without any extra parameters
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello async"}],
            max_tokens=10
        )
        
        print(f"Async response: {response.choices[0].message.content}")
        print("Async API test successful!")
        return True
    except Exception as e:
        print(f"ERROR in async API test: {e}")
        return False

# Run the async test (in a try/except to catch any event loop issues)
try:
    asyncio.run(test_async_openai())
except Exception as e:
    print(f"Error running async test: {e}")

print("\nTest completed.")