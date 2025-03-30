import os


def get_file_extension(filepath):
  _, extension = os.path.splitext(filepath)
  return extension
