import haystack
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore


def main(
    es_retriever,
    item_id: str,
    query: str,
    top_k: int
):

    # item_id = 'B0074BW614'
    # query = 'Is it good for reading?'

    retrieved_docs = es_retriever.retrieve(
        query=query,
        top_k=top_k,
        filters={
            'item_id': [item_id],
            'split': ['test']
        }
    )

    return retrieved_docs

def get_retriever(
    document_store: ElasticsearchDocumentStore
):
    # https://docs.haystack.deepset.ai/docs/retriever
    # from haystack.retriever.sparse import ElasticsearchRetriever
    # es_retriever = ElasticsearchRetriever(document_store=document_store)
    # es_retriever = haystack.nodes.EmbeddingRetriever(document_store=document_store)
    # es_retriever = haystack.nodes.DensePassageRetriever(document_store=document_store)
    es_retriever = haystack.nodes.BM25Retriever(document_store=document_store)
    
    # es_retriever = haystack.nodes.EmbeddingRetriever(
    #     document_store=document_store,
    #     embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    #     model_format="sentence_transformers"
    # )

    # document_store.update_embeddings(es_retriever)

    return es_retriever


if __name__ == '__main__':

    pass
