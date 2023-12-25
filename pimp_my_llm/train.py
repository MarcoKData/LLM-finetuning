import json
from datetime import datetime
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .ChatDataClass import ChatData
from torch.utils.data import DataLoader
from torch.optim import Adam
from .inference import answer_my_question


# Help Functions
def train(chat_data,
model,
tokenizer,
optimizer,
epochs,
device,
path_to_save_model,
print_batch_counter=False,
save_test_results_dest=None,
test_prompt=None,
log_prefix=None):
  t0 = datetime.now()
  for i in range(epochs):
    if save_test_results_dest is not None and test_prompt is not None:
      t1 = datetime.now()
      dt_seconds = (t1 - t0).total_seconds()
      # test model and save result to dest
      if not os.path.exists(save_test_results_dest):
        with open(save_test_results_dest, "w") as file:
          file.write(json.dumps([], indent=4))

      test_result = answer_my_question(test_prompt, model, tokenizer, max_new_tokens=512)

      with open(save_test_results_dest, "r") as file:
        results = json.load(file)      

      entry = {
        "model-path": path_to_save_model,
        "seconds-trained": str(dt_seconds),
        "epochs-trained": str(i),
        "prompt": test_prompt,
        "answer": test_result
      }
      if log_prefix is not None:
        entry["log_prefix"] = log_prefix

      results.append(entry)

      with open(save_test_results_dest, "w") as file:
        file.write(json.dumps(results, indent=4))

    print(f"{i + 1}/{epochs}...")
    total_batches = len(chat_data)
    batch_counter = 1
    for X, a in chat_data:
      if print_batch_counter:
        print(f"Batch {batch_counter}/{total_batches}...")

      batch_counter += 1
      X = X.to(device)
      a = a.to(device)

      optimizer.zero_grad()
      loss = model(X, attention_mask=a, labels=X).loss
      loss.backward()
      optimizer.step()

    torch.save(model.state_dict(), path_to_save_model)
# End Help Functions


# Setup
PATH_PROJECT = os.path.join(".")
PATH_DATA = os.path.join(PATH_PROJECT, "data", "alpaca-dataset.txt")
PATH_MODEL_STATE_SAVE = os.path.join(PATH_PROJECT, "models", "model_state.pt")


def pimp_model(model,
tokenizer,
path_data,
path_save_model,
epochs,
print_batch_counter,
save_test_results_dest,
test_prompt,
log_prefix=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Sending to device", device)
    model = model.to(device)

    # Test Pretrained Model
    prompt_test = "hey i was good at basketball but "
    tokens = tokenizer(prompt_test, return_tensors="pt")
    tokens = tokens.to(device)
    resp_tokens = model.generate(**tokens)
    resp = tokenizer.decode(resp_tokens[0])
    print(f"\n\nUntuned Response to '{prompt_test}':\n{resp}")

    # Load Data
    chat_data = ChatData(path_data, tokenizer)
    chat_data = DataLoader(chat_data, batch_size=64)
    print("Successfully loaded chat data!")

    # Train Model
    optimizer = Adam(model.parameters())

    train(
      chat_data,
      model,
      tokenizer,
      optimizer,
      epochs,
      device,
      path_save_model,
      print_batch_counter,
      save_test_results_dest,
      test_prompt,
      log_prefix
    )

    print("\n\nSuccessfully trained model!")
    print(f"Saved to {path_save_model}")
