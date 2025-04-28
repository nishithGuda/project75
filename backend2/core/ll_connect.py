from huggingface_hub import InferenceClient

# Replace with your actual API key
client = InferenceClient(
    provider="nebius",
    api_key="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"
)

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def call_llama(prompt: str, max_tokens: int = 256) -> str:
    """
    Calls the hosted Mistral model via Hugging Face API with a given prompt.

    Args:
        prompt (str): The natural language prompt for the LLM.
        max_tokens (int): Max tokens to generate.

    Returns:
        str: Response string from Mistral.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error from Mistral API] {e}"
