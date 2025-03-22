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

    corpus_entity = []
    with open(f"expr/{data_source}/vdb_entities.json") as f:
        entities = json.load(f)
        for entity in entities['data']:
            corpus_entity.append(entity['entity_name'])
    
    corpus_hyperedge = []
    with open(f"expr/{data_source}/vdb_hyperedges.json") as f:
        relations = json.load(f)
        for relation in relations['data']:
            corpus_hyperedge.append(relation['hyperedge_name'])

    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        # devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
    )

    embeddings = model.encode_corpus(corpus)
    #save
    np.save(f"expr/{data_source}/corpus.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)
    
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index.bin")
    
    embeddings = model.encode_corpus(corpus_entity)
    #save
    np.save(f"expr/{data_source}/corpus_entity.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_entity.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)
    
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_entity.bin")
    
    embeddings = model.encode_corpus(corpus_hyperedge)
    #save
    np.save(f"expr/{data_source}/corpus_hyperedge.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_hyperedge.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)
    
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_hyperedge.bin")