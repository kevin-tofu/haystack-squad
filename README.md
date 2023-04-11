
# Haystack-Squad-Handler

```bash

poetry install

```

## 

```bash

poetry run python src/evaluate.py --dataset subjqa --dataset_name electronics

```

## Argparsing

parser.add_argument('--elasticsearch_user', '-EU', type=str, default='', help='')
    parser.add_argument('--elasticsearch_password', '-EPW', type=str, default='', help='')
    parser.add_argument('--elasticsearch_port', '-EP', type=int, default=9200, help='')
    parser.add_argument('--elasticsearch_index_document', '-EID', type=str, default='document', help='')
    parser.add_argument('--elasticsearch_index_label', '-EIL', type=str, default='label', help='')

| Args | Example | Description |
| --- | --- | --- |
| dataset | subjqa |  |
| dataset_name | electronics |  |
| elasticsearch_host | localhost |  |
| elasticsearch_password |  |  |
| elasticsearch_port | 9200 |  |
| elasticsearch_index_document | document |  |
| elasticsearch_index_label | label |  |
