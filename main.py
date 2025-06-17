# main.py
import warnings
from retriever import Retriever
from llm_interface import QwenLLM
import re

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added")

retriever = Retriever()
llm = QwenLLM()

# 提取查询主语（关键词）
def extract_main_subject(query):
    subjects = ['owner', '泰山学堂', '山东大学']
    for sub in subjects:
        if sub in query:
            return sub
    return None

# 根据主语过滤检索结果，只保留包含主语的段落
def filter_results_by_subject(results, subject):
    if not subject:
        return results
    filtered = [r for r in results if subject in r['text']]
    return filtered if filtered else results

# 构建简洁上下文，只包含相关片段
def build_context(results, subject):
    filtered = filter_results_by_subject(results, subject)
    lines = []
    for i, r in enumerate(filtered):
        lines.append(f"相关内容 {i+1} (相似度: {r['score']:.2f}): {r['text']}")
    return "".join(lines)

# 若问句含有预设关键词，则优先考虑直接从文档回答
def should_use_local_knowledge(query):
    local_keywords = ['顾九宁', '泰山学堂', '山东大学']
    return any(keyword in query for keyword in local_keywords)

def should_use_rag(results, threshold=0.5):     # 若检索结果中有高于阈值的相似度分数，则使用 RAG 策略
    return bool(results) and any(r['score'] > threshold for r in results)

def get_answer_strategy(query, results):
    if not results:
        return "model"
    if should_use_local_knowledge(query):
        return "local"
    elif should_use_rag(results):
        return "rag"
    else:
        return "model"

print("Welcome to your personal Qwen-RAG assistant!")
while True:
    query = input("You: ")
    if query.lower() in {"exit", "quit", "bye"}:
        break

    # 初次检索
    try:
        results = retriever.retrieve(query, top_k=5)
    except Exception as e:
        print(f"检索出错: {str(e)}")
        results = []

    # 本地知识意图检测和空结果回退
    subject = extract_main_subject(query)
    if should_use_local_knowledge(query) and not results and subject:
        # 从所有文本中抽取包含主语的段落作为本地结果
        results = [
            {"text": t['text'], "score": 1.0}
            for t in retriever.texts if subject in t['text']
        ][:3]

    strategy = get_answer_strategy(query, results)
    print(f"[debug] 策略 = {strategy}, 检索结果数 = {len(results)}")

    if strategy in {"local", "rag"} and results:
        context = build_context(results, subject)
        response = llm.answer(query, context)
    else:
        response = llm.answer(query)

    # 可选后处理
    try:
        post = getattr(llm, '_postprocess_answer', None)
        if callable(post):
            response = post(query, response)
    except Exception:
        pass

    print("Bot:", response)