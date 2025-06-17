from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen2-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype="auto", trust_remote_code=True
        )       #强制cpu推理, 确定性生成

    def answer(self, query, context=""):
        context = context.strip()
        #双prompt模式
        if not context or context == "未找到相关信息":
            system_prompt = (
                "你是一个严谨可靠的问答助手。\n"
                "请直接回答用户的问题，保持简洁、准确。\n"
                "如无法确认答案请明确说明。"
            )
            user_content = query
        else:
            system_prompt = (
                "你是一个严谨可靠的问答助手。\n"
                "请优先使用用户提供的信息进行回答；若信息不足，可补充你掌握的相关知识。\n"
                "请保持简洁、准确，如无法确认答案请明确说明。"
            )
            user_content = f"已知以下信息：\n{context}\n\n请根据以上信息回答：{query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,     #限制生成长度
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in decoded:      #定位assistant标记提取的有效内容
            answer = decoded.split("assistant")[-1].strip(":> \n")
        else:
            answer = decoded.strip()

        return answer if answer else "抱歉，我暂时无法回答这个问题。"   #空回答
#llm_interface.py