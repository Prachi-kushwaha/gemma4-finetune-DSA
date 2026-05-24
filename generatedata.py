import os
import json

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("Open_Router_api_key")
)
QUESTION_TYPES = [
    "intuition",
    "optimized_solution",
    "debugging",
    "dry_run"
]

with open("metadata.json", "r") as f:
    metadata = json.load(f)

dataset = []

for item in tqdm(metadata):

    for qtype in QUESTION_TYPES:

        prompt = f""" you are an expert DSA tutor.
        Topic: {item['topic']}
        Problem:{item['problem']}
        Question type:{qtype}

        Generate:
        1 Explanation
        2 python solutions
        3 Complexity analysis

        Keep response concise and educational

        Return in JSON:
        {{
            "instruction":"...",
            "output":"..."
        }}
        """

        response = client.chat.completions.create(
            model = "google/gemma-4-31b-it:free",
            messages=[
                {"role":"user", "content":prompt}
            ],
            temperature=0.7
        )

        text = response.choices[0].message.content

        try:
            data = json.loads(text)
            dataset.append(data)
        except:
            print("Failed")

with open("generated/dataset.jsonl", "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

print("Done")
