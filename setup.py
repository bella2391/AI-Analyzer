import argparse
import json
import os
import sqlite3
import subprocess
import uuid
from datetime import datetime

import numpy as np

from src.gemini.client import GeminiAPIClient
from src.utils.database import get_embedding_question, get_embeddings, get_file_content
from src.utils.file import get_file_extension
from src.utils.similarity import calculate_cosine_similarity


def make_source_db(target_dir, extension_or_files):
  project_root = os.environ.get("PROJECT_ROOT", ".")
  data_path = os.path.join(project_root, "data")
  unique_id = str(uuid.uuid4())
  db_file_path = os.path.join(data_path, f"{unique_id}.db")
  map_file_path = os.path.join(data_path, "source_db_map.json")

  if not target_dir:
    target_dir = input("Enter the directory path to create db source: ")

  if not extension_or_files:
    choice = input(
      "Which one? [1] files (require file extension) [2] files-list (require comma-separated pair of filenames): "
    )
    if choice == "1":
      extensions = input("Enter the file extensions: ")
      files_arg = f"{target_dir},*.{extensions}"
      file_type = "files"
    elif choice == "2":
      files_list = input("Enter the files with comma-separated: ")
      files_arg = f"{target_dir},{files_list}"
      file_type = "files-list"
    else:
      print("Invalid choice.")
      return
  else:
    files_arg = f"{target_dir},{extension_or_files}"
    file_type = (
      "files-list"  # or "files", depending on how extension_or_files is formatted
    )

  subprocess.run(
    ["gemini-cli", "embed", "db", db_file_path, "--files", files_arg], check=True
  )

  map_entry = {
    "path": target_dir,
    "uuid": unique_id,
    "type": file_type,
    "time": datetime.now().isoformat(),
  }

  if os.path.exists(map_file_path):
    with open(map_file_path, "r") as f:
      existing_map = json.load(f)
      existing_map.append(map_entry)
    with open(map_file_path, "w") as f:
      json.dump(existing_map, f, indent=4)
  else:
    with open(map_file_path, "w") as f:
      json.dump([map_entry], f, indent=4)

  print(f"Saved at {db_file_path}")


def make_question_db(question_text):
  project_root = os.environ.get("PROJECT_ROOT", ".")
  data_path = os.path.join(project_root, "data")
  db_file_path = os.path.join(data_path, "question.db")
  map_file_path = os.path.join(data_path, "question_map.json")

  unique_id = str(uuid.uuid4())
  txt_file_path = os.path.join(data_path, f"{unique_id}.txt")
  with open(txt_file_path, "w") as f:
    f.write(question_text)

  subprocess.run(
    ["gemini-cli", "embed", "db", db_file_path, "--files", txt_file_path], check=True
  )

  map_entry = {
    "question": question_text,
    "uuid": unique_id,
    "time": datetime.now().isoformat(),
  }

  if os.path.exists(map_file_path):
    with open(map_file_path, "r") as f:
      existing_map = json.load(f)
      existing_map.append(map_entry)
    with open(map_file_path, "w") as f:
      json.dump(existing_map, f, indent=4)
  else:
    with open(map_file_path, "w") as f:
      json.dump([map_entry], f, indent=4)

  print(f"Question saved at {db_file_path}")


def select_db(data_path, question_text):
  conn_out = sqlite3.connect(os.path.join(data_path, "common_database.db"))
  cursor_out = conn_out.cursor()
  embeddings_out = get_embeddings(cursor_out)

  conn_question = sqlite3.connect(os.path.join(data_path, "question.db"))
  cursor_question = conn_question.cursor()
  embedding_question = get_embedding_question(cursor_question)

  similarities = []
  for id_value, embedding_out in embeddings_out:
    embedding_list = np.frombuffer(embedding_out, dtype=np.float32)
    similarity = calculate_cosine_similarity(embedding_list, embedding_question)
    similarities.append((similarity, id_value))

  similarities.sort(reverse=True, key=lambda x: x[0])
  most_similar_id = similarities[0][1]

  file_content = get_file_content(cursor_out, most_similar_id)

  api_client = GeminiAPIClient()

  try:
    prompt = f"""Explain the content about below code by Japanese:

            ```{get_file_extension(most_similar_id)}
            {file_content}
            ```
            """
    response_text = api_client.generate_response(prompt)
    print(response_text)
  except Exception as e:
    print(f"Error: failed to request gemini api: {e}")
  finally:
    del api_client


def main():
  parser = argparse.ArgumentParser(description="AI Code Analyzer")
  parser.add_argument(
    "--make-source-db", "-msb", nargs="*", help="Make source database"
  )
  parser.add_argument(
    "--make-question-db",
    "-mqb",
    type=str,
    metavar="question_text",
    help="Make question database",
  )
  parser.add_argument(
    "--select-db",
    "-sb",
    type=str,
    metavar="question_text",
    help="Select database and answer question",
  )
  parser.add_argument("--all", "-a", action="store_true", help="Run all commands")

  args = parser.parse_args()

  project_root = os.environ.get("PROJECT_ROOT", ".")
  data_path = os.path.join(project_root, "data")

  if args.make_source_db:
    target_dir = args.make_source_db[0] if len(args.make_source_db) > 0 else None
    extension_or_files = (
      args.make_source_db[1] if len(args.make_source_db) > 1 else None
    )
    make_source_db(target_dir, extension_or_files)
  elif args.make_question_db:
    make_question_db(args.question_text)
  elif args.select_db:
    select_db(data_path, args.select_db)
  elif args.all:
    pass
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
