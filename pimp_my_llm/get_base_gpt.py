from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_base_gpt2(model_name: str):
    # Create Raw Model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def get_base_llama_7b_hf():
    # Create Raw Model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    return model, tokenizer
