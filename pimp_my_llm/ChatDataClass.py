from torch.utils.data import Dataset


class ChatData(Dataset):
  def __init__(self, path, tokenizer, limit=None):
    with open(path, "r") as file:
      self.X = file.readlines()
    
    if limit is not None:
      self.X = self.X[:limit]

    for i in range(len(self.X)):
      self.X[i] = self.X[i].replace("\n", "").strip()
    
    for i in range(3):
      print(self.X[i])

    self.X_encoded = tokenizer(self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
    self.input_ids = self.X_encoded["input_ids"]
    self.attention_mask = self.X_encoded["attention_mask"]

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return (self.input_ids[idx], self.attention_mask[idx])
