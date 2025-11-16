from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import torch


def create_local_qwen_model():
    model_path = "./llm/qwen-instruct"
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用半精度节省显存
        trust_remote_code=True
    ).to(device)
    
    # 创建 pipeline
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.1,
        do_sample=True,
        top_p=0.9
    )
    
    # 包装成 LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

if __name__ == "__main__":
    llm = create_local_qwen_model()
    print(llm.invoke("你好"))
