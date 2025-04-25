import time

from collections import defaultdict

import lazyllm
from lazyllm import Reranker

from online_models import embedding_model, rerank_model

DOC_PATH = "/mnt/lustre/share_data/dist/cmrc2018/data_kb" 
milvus_store_conf = {
    'type': 'map',
    'indices': {
        'smart_embedding_index': {
        'backend': 'milvus',
        'kwargs': {
            'uri': "dbs/test_cache.db",
            'index_kwargs': {
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            }
        },
        },
    },
}


docs = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model, store_conf=milvus_store_conf)

docs.create_node_group(name='sentence', parent="MediumChunk", transform=(lambda d: d.split('。')))

retriever1 = lazyllm.Retriever(docs, group_name="MediumChunk", topk=6, index='smart_embedding_index')
retriever2 = lazyllm.Retriever(docs, group_name="sentence", target="MediumChunk", topk=6, index='smart_embedding_index')
retriever1.start()
retriever2.start()

reranker = Reranker('ModuleReranker', model=rerank_model, topk=3)

# 设置固定query
query = "证券管理的基本规范？"

# 运行5次没有缓存机制的检索流程，并记录时间
time_no_cache = []
for i in range(5):
    st = time.time()
    nodes1 = retriever1(query=query)
    nodes2 = retriever2(query=query)
    rerank_nodes = reranker(nodes1 + nodes2, query)
    et = time.time()
    t = et - st
    time_no_cache.append(t)
    print(f"No cache 第 {i+1} 次查询耗时：{t}s")

# 定义dict[list]，存储已检索的query和节点集合，实现简易的缓存机制
kv_cache = defaultdict(list)
for i in range(5):
    st = time.time()
    #如果query未在缓存中，则执行正常的检索流程，若query命中缓存，则直接取缓存中的节点集合
    if query not in kv_cache:
        nodes1 = retriever1(query=query)
        nodes2 = retriever2(query=query)
        rerank_nodes = reranker(nodes1 + nodes2, query)
        # 检索完毕后，缓存query及检索节点
        kv_cache[query] = rerank_nodes
    else:
        rerank_nodes = kv_cache[query]
    et = time.time()
    t = et - st
    time_no_cache.append(t)
    print(f"KV cache 第 {i+1} 次查询耗时：{t}s")
