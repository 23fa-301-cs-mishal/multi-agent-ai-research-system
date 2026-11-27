
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: Gemini API key not found in .env")
    raise SystemExit

print("API key loaded successfully.", flush=True)

try:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=15000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )

    print("Sending request...", flush=True)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with just the word HELLO."
    )

    print("SUCCESS:", response.text, flush=True)

except Exception as e:
    print("\nREQUEST FAILED")
    print("Error type:", type(e).__name__)
    print("Error details:", str(e), flush=True)
