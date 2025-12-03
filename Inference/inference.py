import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/pscratch/sd/n/nataraj2/mistral-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Read prompt from file
with open("prompt.txt", "r") as f:
    prompt = f.read()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=3000,
    do_sample=False
)

code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save output to file
with open("output.txt", "w") as f:
    f.write(code)

print("Output saved to output.txt")
