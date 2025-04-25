import time

import lazyllm
from lazyllm import Reranker, LOG

from online_models import embedding_model, llm, rerank_model

DOC_PATH = "/mnt/lustre/share_data/dist/cmrc2018/data_kb" 

milvus_store_conf = {
    'type': 'map',
    'indices': {
        'smart_embedding_index': {
            'backend': 'milvus',
            'kwargs': {
                'uri': "dbs/test_map_milvus.db",
                'index_kwargs': {
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                }
            },
        },
    },
}

docs = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model, store_conf=milvus_store_conf)

# 使用默认索引
retriever1 = lazyllm.Retriever(docs, group_name="MediumChunk", topk=6, similarity="cosine")
# 使用smart_embedding_index索引，调用milvus向量检索
retriever2 = lazyllm.Retriever(docs, group_name="MediumChunk", topk=6, index='smart_embedding_index')
retriever1.start()
retriever2.start()

reranker = Reranker('ModuleReranker', model=rerank_model, topk=3)

prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
        根据以下资料回答问题：\
        {context_str} \n '
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

query = "证券管理有哪些准则？"

nodes1 = retriever1(query=query)
nodes2 = retriever2(query=query)
rerank_nodes = reranker(nodes1 + nodes2, query)

context_str = "\n======\n".join([node.get_content() for node in rerank_nodes])
res = llm({"query": query, "context_str": context_str})
print(res)
