def generate_gemini_response(client, model_id, prompt):
  return client.models.generate_content(model=model_id, contents=prompt)
