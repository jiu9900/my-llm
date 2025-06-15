# 要求
(1）部署一个通义干问1.5B、4B，或者 DEEPSEEK 1.5B、7B的模型，如果没有显卡，尝试使用 CPU 部署；(2）能够进行基本的短文本问答；
额外内容：(1）制作个人知识库，对部署的模型进行 RAG ，能够输入自己的姓名等，进行正确的问答；(2）使用微调等其他手段，提高模型对于个人、对于山东大学的回答正确率。

# 6/12
已实现:
1. 通义千问1.5B的CPU部署（llm_interface.py）

2. RAG检索系统（retriever.py）

3. 个人知识库（gjn.txt）

# 6/15
实现实现自主判断使用 RAG、本地知识 or 自带知识
引入moka-ai/m3e-base帮助文本相似度检索
Retriever用的是moka-ai/m3e-base，这是句向量模型（SentenceTransformer），用来做文本相似度检索的。
QwenLLM是一个独立的语言模型（大语言模型），加载的是Qwen/Qwen2-1.5B，用来做问答生成的。
本地知识库的召回靠moka-ai/m3e-base做文本编码，回答生成靠Qwen模型