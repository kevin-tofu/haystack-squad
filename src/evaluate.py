import argparse
import uuid
from typing import Optional
import haystack
# import datasets
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from haystack import Document, Label#, Span, Answer

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever import BaseRetriever


# https://github.com/nlp-with-transformers/notebooks/issues/50

def evaluate_retriever(
    document_store: ElasticsearchDocumentStore,
    es_retriever: BaseRetriever,
    elasticsearch_index_label: str
):

    # https://docs.haystack.deepset.ai/docs/documents_answers_labels
    # https://docs.haystack.deepset.ai/reference/document-store-api#elasticsearchdocumentstore

    labels_agg = document_store.get_all_labels_aggregated(
        index=elasticsearch_index_label,
        open_domain=True,
        aggregate_by_meta=["item_id"]
    )

    pipe_retrieval = DocumentSearchPipeline(es_retriever)
    eval_result = pipe_retrieval.eval(labels=labels_agg, params={"Retriever": {"top_k": 10}})
    metrics = eval_result.calculate_metrics()

    return metrics, eval_result


if __name__ == '__main__':

    # domains = datasets.get_dataset_config_names('subjqa')
    # subjqa = datasets.load_dataset('subjqa', name='electronics')
    
    import utils
    import es
    import retrieve
    
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
    
    retriever = retrieve.get_retriever(
        documentstore   
    )

    metrics, eval_result = evaluate_retriever(
        documentstore,
        retriever,
        args.elasticsearch_index_label
    )

    print(type(documentstore))
    print(type(retriever))

    print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
    print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
    print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
    print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
    print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')
    # print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
    # print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')
