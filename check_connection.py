"""Quick check that your .env settings reach Azure OpenAI.   python check_connection.py"""
import os
import sys

from openai import APIConnectionError, APIStatusError

from rag import embed, get_client


def _hint(exc: Exception) -> str:
    code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if code == 401 or "invalid subscription key" in text or "access denied" in text:
        return "Auth failed (401): verify AZURE_OPENAI_API_KEY and endpoint belong to the same Azure OpenAI resource."
    if code == 404 or "resource not found" in text:
        return "Not found (404): verify endpoint format and both deployment names are exact matches from Azure AI Foundry."
    if isinstance(exc, APIConnectionError) or "connection error" in text:
        return "Connection failed: check AZURE_OPENAI_ENDPOINT and your network access."
    return "Check endpoint, key, API version, and deployment names in .env."


try:
    client = get_client()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    chat_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

    print(f"Endpoint: {endpoint}")
    print(f"Embeddings deployment: {embedding_deployment}")
    print(f"Chat deployment: {chat_deployment}")

    try:
        vec = embed(["connection test"])
        print(f"OK  embeddings deployment '{embedding_deployment}' (vector size {vec.shape[1]})")
    except Exception as e:
        print(f"FAILED at embeddings call: {e}")
        print(_hint(e))
        sys.exit(1)

    try:
        resp = client.chat.completions.create(
            model=chat_deployment,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        print(f"OK  chat deployment '{chat_deployment}' replied: "
              f"{(resp.choices[0].message.content or '').strip()}")
    except Exception as e:
        print(f"FAILED at chat call: {e}")
        print(_hint(e))
        sys.exit(1)

    print("SUCCESS: Endpoint, key, and both deployments are working.")
except Exception as e:
    print(f"FAILED: {e}")
    if isinstance(e, APIStatusError):
        print(f"HTTP status: {e.status_code}")
    print(_hint(e))
    sys.exit(1)
