import json
import numpy as np
# encode
from FlagEmbedding import FlagAutoModel
import faiss
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', default='2wikimultihopqa')
    args = parser.parse_args()
    data_source = args.data_source
    
    corpus = []
    with open(f"datasets/{data_source}/corpus.jsonl") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["contents"])


    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        # devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
    )

    embeddings = model.encode_corpus(corpus)
    #save
    np.save(f"datasets/{data_source}/corpus.npy", embeddings)

    corpus_numpy = np.load(f"datasets/{data_source}/corpus.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)
    
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"datasets/{data_source}/index.bin")