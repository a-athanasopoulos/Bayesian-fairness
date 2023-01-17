import os


def create_directory(path):
    """
    create directory
    :param path:
    :return:
    """
    if not os.path.isdir(path):
       os.makedirs(path)