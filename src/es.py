
from haystack import Document, Label
from haystack.document_stores import ElasticsearchDocumentStore

def list2documentstore(
    elasticsearch_host: str,
    elasticsearch_port: int,
    elasticsearch_user: str,
    elasticsearch_password: str,
    elasticsearch_index_docs: str,
    elasticsearch_index_labels: str,
    docs: list[Document],
    labels: list[Label]
):
    document_store = ElasticsearchDocumentStore(
        host=elasticsearch_host,
        port=elasticsearch_port,
        username=elasticsearch_user,
        password=elasticsearch_password,
        # index=elasticsearch_index,
        # embedding_field="question_emb",
        # embedding_dim=384,
        return_embedding=True
    )

    # create aggregated labels
    document_store.write_documents(docs, index=elasticsearch_index_docs)
    document_store.write_labels(labels, index=elasticsearch_index_labels)
    
    return document_store


if __name__ == '__main__':

    import utils
    import es
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='subjqa', help='')
    parser.add_argument('--dataset_name', '-DN', type=str, default='electronics', help='')
    parser.add_argument('--elasticsearch_host', '-EH', type=str, default='localhost', help='')
    parser.add_argument('--elasticsearch_user', '-EU', type=str, default='', help='')
    parser.add_argument('--elasticsearch_password', '-EPW', type=str, default='', help='')
    parser.add_argument('--elasticsearch_port', '-EP', type=int, default=9200, help='')
    parser.add_argument('--elasticsearch_index_document', '-EID', type=str, default='document', help='')
    parser.add_argument('--elasticsearch_index_label', '-EIL', type=str, default='label', help='')
    # parser.add_argument('--elasticsearch_tokenizer', '-ET', type=str, default='kuromoji_tokenizer', help='')
    # parser.add_argument('--path_export', '-PE', type=str, default='./output', help='')
    args = parser.parse_args()


    dfs = utils.get_dataset(args)
    docs, labels = utils.df2docslabels(dfs['test'])

    documentstore = es.list2documentstore(
        args.elasticsearch_host,
        args.elasticsearch_port,
        args.elasticsearch_user,
        args.elasticsearch_password,
        args.elasticsearch_index_document,
        args.elasticsearch_index_label,
        docs,
        labels
    )
