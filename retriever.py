# retriever.py
from sentence_transformers import SentenceTransformer, util
import re
import pickle
import hashlib
import os


def file_md5(path):     #读取整个文件内容，计算 MD5 哈希。用于判断文本是否变化
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


class Retriever:
    def __init__(self, text_path="docs/gjn.txt", model_name="moka-ai/m3e-base"):    #M3E中文嵌入模型
        self.model = SentenceTransformer(model_name)
        self.text_path = text_path
        self.hash = file_md5(text_path)
        self.pkl_path = 'embeddings.pkl'

        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'rb') as f:        # 若本地 embeddings.pkl 存在且保存的哈希与当前文件一致，则直接加载上次构建的段落列表与对应张量，避免重新编码。
                cache = pickle.load(f)
                if cache.get('hash') == self.hash:
                    print("[cache] 加载嵌入向量缓存")
                    self.texts = cache['texts']
                    self.embeddings = cache['embeddings']
                    return

        print("[cache] 没有命中缓存，重新构建向量")
        self.texts = self._load_and_chunk(text_path)        #将全文切分成若干“主题段落”；
        self.embeddings = self.model.encode(
            [t['text'] for t in self.texts], convert_to_tensor=True
        )
        with open(self.pkl_path, 'wb') as f:
            pickle.dump({
                'hash': self.hash,
                'texts': self.texts,
                'embeddings': self.embeddings
            }, f)

    def _load_and_chunk(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # 使用自然语言分段，以句号或换行分割，保证段落聚焦主题
        paragraphs = re.split(r'(?<=[。！？])|+', full_text.strip())

        return [
            {"text": p.strip(), "meta": {"source": path, "index": i}}
            for i, p in enumerate(paragraphs) if p.strip()
        ]

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]      #余弦相似度计算
        top_results = scores.topk(k=top_k)      #使用 PyTorch 的 topk 函数获取最大值索引

        real_top_k = min(top_k, len(self.texts))
        return [
            {"text": self.texts[idx]['text'], "score": float(scores[idx])}
            for idx in top_results.indices
        ]

from sentence_transformers import SentenceTransformer, util
import re
import pickle
import hashlib
import os


def file_md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


class Retriever:
    def __init__(self, text_path="docs/gjn.txt", model_name="moka-ai/m3e-base"):
        self.model = SentenceTransformer(model_name)
        self.text_path = text_path
        self.hash = file_md5(text_path)
        self.pkl_path = 'embeddings.pkl'

        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'rb') as f:
                cache = pickle.load(f)
                if cache.get('hash') == self.hash:
                    print("[cache] 加载嵌入向量缓存")
                    self.texts = cache['texts']
                    self.embeddings = cache['embeddings']
                    return

        print("[cache] 没有命中缓存，重新构建向量")
        self.texts = self._load_and_chunk(text_path)
        self.embeddings = self.model.encode(
            [t['text'] for t in self.texts], convert_to_tensor=True
        )
        with open(self.pkl_path, 'wb') as f:
            pickle.dump({
                'hash': self.hash,
                'texts': self.texts,
                'embeddings': self.embeddings
            }, f)

    def _load_and_chunk(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # 使用自然语言分段，以句号或换行分割，保证段落聚焦主题
        paragraphs = re.split(r'(?<=[。！？])|\n+', full_text.strip())

        return [
            {"text": p.strip(), "meta": {"source": path, "index": i}}
            for i, p in enumerate(paragraphs) if p.strip()
        ]

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = scores.topk(k=top_k)

        return [
            {"text": self.texts[idx]['text'], "score": float(scores[idx])}
            for idx in top_results.indices
        ]