import pimp_my_llm as pml
import os


PATH_DATA = os.path.join(".", "data", "alpaca-dataset.txt")
PATH_SAVE_MODEL = os.path.join(".", "models", "model_state_refactored.pt")

model, tokenizer = pml.get_base_gpt2("gpt2-medium")

pml.pimp_model(
    model=model,
    tokenizer=tokenizer,
    path_data=PATH_DATA,
    path_save_model=PATH_SAVE_MODEL,
    epochs=1,
    log_prefix="Test"
)


"""
model = pml.load_weights(model, PATH_SAVE_MODEL)

answer = pml.answer_my_question(
    question="How are you today?",
    model=model,
    tokenizer=tokenizer
)
print("Answer:", answer)"""
