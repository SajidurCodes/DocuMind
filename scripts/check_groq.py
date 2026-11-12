from pathlib import Path

from dotenv import load_dotenv
from groq import Groq



PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
   
    with Groq(timeout=30.0, max_retries=0) as client:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "In one short sentence, explain what an API is.",
                }
            ],
            reasoning_effort="low",
            max_completion_tokens=512,
        )

        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("Groq returned no answer text.")

        print("Groq response:")
        print(answer)


if __name__ == "__main__":
    main()