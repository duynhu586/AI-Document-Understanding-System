
import json
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def parse_document(text):

    prompt = f"""
Extract structured information from the receipt text.

Return JSON with fields:
vendor
date
total

Rules:
- Total must be a valid number
- Remove currency symbols like $, €

Receipt text:
{text}

Return ONLY JSON.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stop=None
    )

    output = response.choices[0].message.content

    try:
        return json.loads(output)
    except:
        return {"raw_output": output}