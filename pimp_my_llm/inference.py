import torch


def answer_my_question(question, model, tokenizer):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)

  inp_in_template = f"{question} <bot>: "
  tokens = tokenizer(inp_in_template, return_tensors="pt")
  X = tokens["input_ids"].to(device)
  a = tokens["attention_mask"].to(device)
  output = model.generate(X, attention_mask=a, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)[0]

  output = tokenizer.decode(output)

  output_cleaned = output.replace(question, "").replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "").replace("<bot>:", "").replace("<pad>", "").replace("'", "").strip()

  if output_cleaned == "":
    output_cleaned = "## Could not generate response ##"

  return output_cleaned


def load_weights(model, path_to_state):
  model.load_state_dict(torch.load(path_to_state))

  return model
