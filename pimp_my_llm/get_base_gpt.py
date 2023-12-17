from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_base_gpt2():
    # Create Raw Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
