import os

from google import genai

MODEL_ID = "gemini-2.0-flash"


class GeminiAPIClient:
  def __init__(self, api_key=None, model_id="gemini-2.0-flash"):
    if api_key is None:
      api_key = os.environ.get("GEMINI_API_KEY")
      if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    self.api_key = api_key
    self.model_id = model_id
    self.client = genai.Client(api_key=api_key)

  def generate_response(self, prompt):
    try:
      response = self.client.models.generate_content(model=MODEL_ID, contents=prompt)
      return response.text
    except Exception as e:
      raise RuntimeError(f"Failed to request Gemini API: {e}")
