import json
import os
import sqlite3
import sys

import numpy as np
from google import genai
from IPython.display import Markdown


def calculate_cosine_similarity(vector1, vector2):
  return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# gemini eat source db file
conn_out = sqlite3.connect("fmc_common.db")
cursor_out = conn_out.cursor()
cursor_out.execute("SELECT embedding FROM embeddings")
embeddings_out = cursor_out.fetchall()

# read question
conn_question = sqlite3.connect("question.db")
cursor_question = conn_question.cursor()
cursor_question.execute("SELECT embedding FROM embeddings")

try:
  # embedding_question = cursor_question.fetchone()[0]
  embedding_question = json.loads(cursor_question.fetchone()[0])
except json.JSONDecodeError as e:
  print(f"Error: JSON decoding failed: {e}")
  sys.exit(1)

# calculate cosine similarity
similarities = []
for embedding_out in embeddings_out:
  try:
    embedding_list = json.loads(embedding_out[0])
    similarity = calculate_cosine_similarity(embedding_list, embedding_question)
    similarities.append(similarity)
  except json.JSONDecodeError as e:
    print(f"Warning: JSON decoding failed for one embedding: {e}")
    continue

cursor_out.execute("SELECT content FROM embeddings ORDER BY similarity DESC LIMIT 1")
code_snippet = cursor_out.fetchone()[0]

print(code_snippet)

# check existence of environment values
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
  print("Error: Not set $env:GEMINI_API_KEY")
  exit()

# see docs: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb
MODEL_ID = "gemini-2.0-flash"
client = genai.Client(api_key=api_key)

prompt = f"What do this code? How do it function?: {code_snippet}"

try:
  response = client.models.generate_content(model=MODEL_ID, contents=prompt)
  Markdown(response.text)
except Exception as e:
  print(f"Error: failed to request gemini api: {e}")
