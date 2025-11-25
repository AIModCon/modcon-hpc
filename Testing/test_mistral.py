import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/pscratch/sd/n/nataraj2/mistral-7b"   # adjust if needed

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Give AMReX C++ code for a 2D Poisson equation with Neumann boundary conditions. Output only code."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# <-- replace your old generate call with this
outputs = model.generate(
    **inputs,
    max_new_tokens=3000,   # increase to get full file
    do_sample=False         # deterministic, less chance of early stopping
)

code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save to a file
with open("poisson.cpp", "w") as f:
    f.write(code)

print("Code saved to poisson.cpp")

