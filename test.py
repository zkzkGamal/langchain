import google.generativeai as genai
import os

# --- Configuration ---
# Replace "YOUR_API_KEY_HERE" with your actual API key.
# It's highly recommended to use environment variables for security.
# For example, you can set an environment variable named GOOGLE_API_KEY.
# If you're using Colab, you can add it to the secrets manager.
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# If you don't want to use environment variables, you can hardcode it (not recommended for production):
# genai.configure(api_key="YOUR_API_KEY_HERE")

# For demonstration purposes, let's assume you've set the API key as an environment variable
# or are running this in an environment where it's already configured.
# If not, uncomment and set your API key.
# genai.configure(api_key="YOUR_API_KEY_HERE") # <-- Replace with your actual API key or use environment variable

# --- List Models ---
def list_available_models():
    """Lists all available generative models and their capabilities."""
    try:
        print("Fetching available models...")
        # genai.list_models() returns an iterable of model objects
        for model in genai.list_models():
            print(f"Model Name: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print(f"  Supported Generation Methods: {model.supported_generation_methods}")
            print("-" * 20)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your API key is correctly configured and has access to the Gemini API.")
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    # You must configure your API key before calling genai.list_models()
    # Example using environment variable:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set your API key or replace os.getenv('GOOGLE_API_KEY') with your actual key.")
    else:
        genai.configure(api_key=api_key)
        list_available_models()