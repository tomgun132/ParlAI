"""Fastext Common Crawl vectors, e.g. use with filename
"models:fasttext_cc_vectors/crawl-300d-2M.vec"
"""
import os
import torchtext.vocab as vocab
import importlib

URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz'

def model_path(datapath, path):
    if path is None:
        return None

    # module_name = 'parlai.projects.jp_dialogue.{}'.format(path)

    # try:
    #     my_module = importlib.import_module(module_name)
    #     my_module.download(datapath)
    # except (ImportError, AttributeError):
    #     pass
    return os.path.join(datapath, 'models', path)

def download(datapath):
    return vocab.Vectors(
        name='cc.ja.300.vec.gz',
        url=URL,
        cache=model_path(datapath, 'fasttext_cc_jp_vectors')
    )
