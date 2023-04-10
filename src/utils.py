import argparse
import datasets
import pandas
from haystack import Document, Label, Span, Answer
# from typing import Optional

# https://github.com/nlp-with-transformers/notebooks/issues/50


def get_dataset(
    args: argparse.Namespace
)-> dict[pandas.core.frame.DataFrame]:

    # domains = datasets.get_dataset_config_names('subjqa')
    # subjqa = datasets.load_dataset('subjqa', name='electronics')
    domains = datasets.get_dataset_config_names(args.dataset)
    subjqa = datasets.load_dataset('subjqa', name=args.dataset_name)
    dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
    # print(domains)

    return dfs


def df2docslabels(
    data: pandas.core.frame.DataFrame
) -> tuple[list[Document], list[Label]]:
   
    # create labels
    labels = list()
    docs = list()
    for i, row in data.iterrows():

        doc = Document(
            id=i,
            content=row["context"],
            content_type="text",
            meta=dict(
                item_id=row['title'],
                split='test',
                question_id=row['id'],
                question=row['question']
            )
        )
        docs.append(doc)

        # Metadata used for filtering in the Retriever
        meta = dict(
            item_id=row["title"],
            split='test',
            question_id=row["id"]
        )

        # Populate labels for questions with answers
        if len(row["answers.text"]):
            for ii, answer in enumerate(row["answers.text"]):
                span_start = row["answers.answer_start"][ii]
                span_end = span_start + len(answer)
                
                ans = Answer(
                    answer=answer,
                    type='extractive',
                    offsets_in_context=[Span(span_start, span_end)]
                )
                
                label = Label(
                    # id=i,
                    # id=str(uuid.uuid4()),
                    query=row["question"],
                    answer=ans,
                    document=doc,
                    origin="gold-label",
                    meta=meta,
                    is_correct_answer=True,
                    is_correct_document=True,
                    no_answer=False
                )
                
                labels.append(label)

        # Populate labels for questions without answers
        else:
            label = Label(
                # id=i,
                # id=str(uuid.uuid4()),
                query=row["question"],
                answer=None,
                document=doc,
                origin="gold-label",
                meta=meta,
                is_correct_answer=True,
                is_correct_document=True,
                no_answer=True
            )
            
            labels.append(label)

    return docs, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='subjqa', help='')
    parser.add_argument('--dataset_name', '-DN', type=str, default='electronics', help='')
    args = parser.parse_args()

    dfs = get_dataset(args)
    print(type(dfs['test']))

    docs, labels = df2docslabels(dfs['test'])
    print('len(docs): ', len(docs))
    print('len(labels): ', len(labels))