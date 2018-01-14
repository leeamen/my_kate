#coding:utf8
from __future__ import absolute_import
import argparse
import math
import numpy as np

from autoencoder.core.ae import load_ae_model
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm, revdict, unitmatrix
from autoencoder.utils.io_utils import dump_json, write_file
from autoencoder.testing.visualize import word_cloud


def gen_docs(args):
    #corpus = load_corpus("./data/20news/output/test.corpus")
    corpus = load_corpus(args.input_corpus)
    vocab, docs = corpus['vocab'], corpus['docs']
    n_vocab = len(vocab)

    #new docs
    new_docs = {}

    for doc_key in docs.keys():
        if doc_key.startswith(args.startswith):
            new_docs[doc_key] = docs[doc_key]
        
    print("{},共有文档:{}".format(args.startswith, len(new_docs)))

    dump_json({"vocab": vocab, "docs": new_docs}, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_corpus', type=str, required=True, help='需要提取的原始词库')
    parser.add_argument('-start', '--startswith', type=str, required=True, help='需要生成得测试集词库')
    parser.add_argument('-o', '--output', type=str, required=True, help='词库文件路径')
    args = parser.parse_args()

    gen_docs(args)

