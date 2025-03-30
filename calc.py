import json
import os
import sqlite3
import sys

import numpy as np
from google import genai


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

row = cursor_question.fetchone()
if row is None:
  print("Error: No data found in question.db")
  sys.exit(1)

embedding_raw = row[0]

if isinstance(embedding_raw, bytes):
  print("Detected binary data, converting to NumPy array...")
  try:
    embedding_question = np.frombuffer(embedding_raw, dtype=np.float32)
  except Exception as e:
    print(f"Error: Failed to convert binary to NumPy array: {e}")
    sys.exit(1)
else:
  print("Error: Expected binary data, but got string.")
  sys.exit(1)

# calculate cosine similarity
similarities = []
for id_value, embedding_out in embeddings_out:
  try:
    embedding_list = np.frombuffer(embedding_out, dtype=np.float32)
    similarity = calculate_cosine_similarity(embedding_list, embedding_question)
    similarities.append((similarity, id_value))
  except json.JSONDecodeError as e:
    print(f"Warning: JSON decoding failed for one embedding: {e}")
    continue

if not similarities:
  print("Error: No valid embedding found.")
  sys.exit(1)

similarities.sort(reverse=True, key=lambda x: x[0])
most_similar_id = similarities[0][1]

print(f"Most similar embedding ID: {most_similar_id}")

# check existence of environment values
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
  print("Error: Not set $env:GEMINI_API_KEY")
  sys.exit(1)

# see docs: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb
MODEL_ID = "gemini-2.0-flash"

prompt = f"What does this code do? How does it function?: {code_snippet}"
client = genai.Client(api_key=api_key)

try:
  response = client.models.generate_content(model=MODEL_ID, contents=prompt)
  print(response.text)
except Exception as e:
  print(f"Error: failed to request gemini api: {e}")
finally:
  del client
