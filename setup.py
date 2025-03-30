import argparse
import os
import sqlite3

import numpy as np

from src.gemini.client import GeminiAPIClient
from src.utils.database import get_embedding_question, get_embeddings, get_file_content
from src.utils.file import get_file_extension
from src.utils.similarity import calculate_cosine_similarity


def make_source_db(target_dir, extension_or_files):
  # gemini-cli を使用してソースデータベースを作成する処理
  pass


def make_question_db(question_text):
  # 質問テキストをデータベースに保存する処理
  pass


def select_db(data_path, question_text):
  # データベースを選択し、質問に対する回答を生成する処理
  conn_out = sqlite3.connect(os.path.join(data_path, "fmc_common_database.db"))
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
    "--make-source-db",
    "-msb",
    nargs=2,
    metavar=("target_dir", "extension_or_files"),
    help="Make source database",
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
    make_source_db(args.make_source_db[0], args.make_source_db[1])
  elif args.make_question_db:
    make_question_db(args.question_text)
  elif args.select_db:
    select_db(data_path, args.select_db)
  elif args.all:
    # すべてのコマンドを実行する処理
    pass
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
