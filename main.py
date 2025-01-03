import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from openai import AuthenticationError, APIConnectionError, RateLimitError, OpenAIError
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.readers import SimpleDirectoryReader

# 환경 변수 로드
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
openai_api_key = os.getenv("OPENAI_API_KEY")


def load_claims(claim_file):
    try:
        with open(claim_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{claim_file}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in '{claim_file}'.")
        exit(1)


def create_embedding(text):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        return embedding
    except AuthenticationError as e:
        print(f"Authentication Error: {e}")
        exit(1)
    except APIConnectionError as e:
        print(f"API Connection Error: {e}")
        exit(1)
    except RateLimitError as e:
        print(f"Rate Limit Exceeded: {e}")
        exit(1)
    except OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        exit(1)


def createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K):
    documents = SimpleDirectoryReader(input_files=[REPORT]).load_data()
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)

    for node in nodes:
        node.embedding = create_embedding(node.text)

    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    return retriever


def demo_agent(context, claim, supplement):
    prompt = {
        "context": context,
        "claim": claim,
        "supplementary": supplement
    }

    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url=f"{deepseek_base_url}/chat/completions",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a fact-checking assistant."},
                    {"role": "user", "content": json.dumps(prompt)}
                ]
            }
        )
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise ValueError("Invalid response format from DeepSeek API")
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Response Error: {e}")
        exit(1)


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <Claim ID>")
        exit(1)

    claim_id = sys.argv[1]
    claim_file = "claim.json"
    pdf_file = "./pdf/1.pdf"

    if not os.path.isfile(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found.")
        exit(1)

    claims = load_claims(claim_file)
    claim_data = claims.get(claim_id)
    if not claim_data:
        print(f"Error: Claim ID '{claim_id}' not found in '{claim_file}'.")
        exit(1)

    claim_text = claim_data["claim"]
    supplement = claim_data.get("supplement", "No supplementary information provided.")

    retriever = createRetriever(
        REPORT=pdf_file,
        CHUNK_SIZE=512,
        CHUNK_OVERLAP=50,
        TOP_K=5
    )
    relevant_nodes = retriever.retrieve(query=claim_text)
    context = "\n".join(node.text for node in relevant_nodes)

    result = demo_agent(context, claim_text, supplement)
    print(f"Claim: {claim_text}\n")
    print(f"Fact-check Result: {result}")


if __name__ == "__main__":
    main()
