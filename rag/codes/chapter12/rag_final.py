from collections import defaultdict

import lazyllm

from online_models import embedding_model, rerank_model, llm

DOC_PATH = "/mnt/lustre/share_data/dist/cmrc2018/data_kb"
milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': "dbs/test_rag.db",
        'index_kwargs': {
        'index_type': 'HNSW',
        'metric_type': 'COSINE',
        }
    }
}
# 定义kv缓存
kv_cache = defaultdict(list)

docs1 = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model, store_conf=milvus_store_conf)
docs1.create_node_group(name='sentence', parent="MediumChunk", transform=(lambda d: d.split('。')))

prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
    根据以下资料回答问题：\
    {context_str} \n '

with lazyllm.pipeline() as recall:
    # 并行多路召回
    with lazyllm.parallel().sum as recall.prl:
        recall.prl.r1 = lazyllm.Retriever(docs1, group_name="MediumChunk", topk=6)
        recall.prl.r2 = lazyllm.Retriever(docs1, group_name="sentence", target="MediumChunk", topk=6)
    recall.reranker = lazyllm.Reranker(name='ModuleReranker',model=rerank_model, topk=3) | lazyllm.bind(query=recall.input)
    recall.cache_save = (lambda nodes, query: (kv_cache.update({query: nodes}) or nodes)) | lazyllm.bind(query=recall.input)
    
with lazyllm.pipeline() as ppl:
    # 缓存检查
    ppl.cache_check = lazyllm.ifs(
        cond=(lambda query: query in kv_cache),
        tpath=(lambda query: kv_cache[query]),
        fpath=recall
    )
    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="\n".join(node.get_content() for node in nodes),
            query=query)
    ) | lazyllm.bind(query=ppl.input)
    ppl.llm = llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

w = lazyllm.WebModule(ppl, port=23492, stream=True).start().wait()
