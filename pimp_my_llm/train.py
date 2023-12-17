import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .ChatDataClass import ChatData
from torch.utils.data import DataLoader
from torch.optim import Adam


# Help Functions
def train(chat_data, model, optimizer, epochs, device, path_to_save_model, print_batch_counter=False):
  for i in range(epochs):
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


def pimp_model(model, tokenizer, path_data, path_save_model, epochs):
    # Test Pretrained Model
    prompt_test = "hey i was good at basketball but "
    tokens = tokenizer(prompt_test, return_tensors="pt")
    resp_tokens = model.generate(**tokens)
    resp = tokenizer.decode(resp_tokens[0])
    print(f"\n\nUntuned Response to '{prompt_test}':\n{resp}")

    # Load Data
    chat_data = ChatData(path_data, tokenizer)
    chat_data = DataLoader(chat_data, batch_size=64)
    print("Successfully loaded chat data!")

    # Train Model
    optimizer = Adam(model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Sending to device", device)
    model = model.to(device)

    train(chat_data, model, optimizer, epochs, device, path_save_model)

    print("\n\nSuccessfully trained model!")
    print(f"Saved to {path_save_model}")
