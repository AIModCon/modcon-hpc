import json

# Input text file
input_file = "amrex_inst_resp_pairs.txt"
# Output JSONL file
output_file = "amrex_dataset.jsonl"

dataset = []

with open(input_file, "r") as f:
    lines = f.readlines()

instruction = None
response_lines = []
for line in lines:
    line = line.rstrip()  # remove trailing newline
    if line.startswith("instruction:"):
        # Save previous pair if exists
        if instruction is not None:
            response_text = "\n".join(response_lines).strip()
            dataset.append({
                "instruction": instruction,
                "response": response_text
            })
            response_lines = []

        # Get current instruction
        instruction = line[len("instruction:"):].strip()
    elif line.startswith("response:"):
        # response starts next lines
        response_lines = []
    else:
        # Collect response lines
        response_lines.append(line)

# Add the last pair
if instruction is not None:
    response_text = "\n".join(response_lines).strip()
    dataset.append({
        "instruction": instruction,
        "response": response_text
    })

# Write to JSONL
with open(output_file, "w") as f:
    for entry in dataset:
        json_line = json.dumps(entry)
        f.write(json_line + "\n")

print(f"Converted {len(dataset)} pairs to {output_file}")

