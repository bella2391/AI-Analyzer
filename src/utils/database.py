import sys

import numpy as np


def get_embeddings(cursor):
  cursor.execute("SELECT id, embedding FROM embeddings")
  return cursor.fetchall()


def get_embedding_question(cursor):
  cursor.execute("SELECT embedding FROM embeddings")
  row = cursor.fetchone()
  if row is None:
    print("Error: No data found in question.db")
    sys.exit(1)

  embedding_raw = row[0]

  if isinstance(embedding_raw, bytes):
    print("Detected binary data, converting to NumPy array...")
    try:
      return np.frombuffer(embedding_raw, dtype=np.float32)
    except Exception as e:
      print(f"Error: Failed to convert binary to NumPy array: {e}")
      sys.exit(1)
  else:
    print("Error: Expected binary data, but got string.")
    sys.exit(1)


def get_file_content(cursor, most_similar_id):
  cursor.execute("SELECT embedding FROM embedding WHERE id = ?", (most_similar_id,))
  row = cursor.fetchone()

  if row is None:
    print(f"Error: No content found for ID {most_similar_id}")
    sys.exit(1)

  return row[0].decode("utf-8")
