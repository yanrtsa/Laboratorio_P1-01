import json
import random
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

domain = "cooking recipes"

def generate_pairs(num_pairs=10):
    prompt = f"""
Generate {num_pairs} pairs of prompts and responses in the domain of {domain}.
Each pair should be a question/instruction about cooking and its expected answer.
Format the output as a valid JSON list of objects, where each object has "prompt" and "response" keys.
Example:
[
    {{
        "prompt": "How to make spaghetti carbonara?",
        "response": "To make spaghetti carbonara, boil pasta, cook pancetta, mix eggs and cheese, combine everything."
    }},
    ...
]
Ensure the JSON is valid and contains exactly {num_pairs} items.
"""
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        pairs = json.loads(text)
        return pairs
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response text: {text}")
        return []

all_pairs = []
batch_size = 10
num_batches = 5

for i in range(num_batches):
    print(f"Generating batch {i+1}/{num_batches}")
    pairs = generate_pairs(batch_size)
    all_pairs.extend(pairs)
    if len(pairs) < batch_size:
        print(f"Warning: Only got {len(pairs)} pairs in batch {i+1}")

# Ensure we have at least 50
if len(all_pairs) < 50:
    print(f"Only generated {len(all_pairs)} pairs, trying to get more")
    additional = generate_pairs(50 - len(all_pairs))
    all_pairs.extend(additional)

random.shuffle(all_pairs)

train_size = int(0.9 * len(all_pairs))
train_data = all_pairs[:train_size]
test_data = all_pairs[train_size:]

with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('test.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Generated {len(all_pairs)} pairs. Train: {len(train_data)}, Test: {len(test_data)}")
print("Saved to train.json and test.json")