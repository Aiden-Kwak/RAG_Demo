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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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


def load_document_metadata(info_file):
    try:
        with open(info_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file '{info_file}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in '{info_file}'.")
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
    except (AuthenticationError, APIConnectionError, RateLimitError, OpenAIError) as e:
        print(f"OpenAI API Error: {e}")
        exit(1)


def createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, info_file):
    documents = SimpleDirectoryReader(input_files=[REPORT]).load_data()
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)

    document_metadata = load_document_metadata(info_file)
    document_id = os.path.basename(REPORT).split(".")[0]  # 아이디랑 문서명이랑 통일했음

    for node in nodes:
        if document_id in document_metadata:
            node.metadata = {
                "document_name": document_metadata[document_id]["title"],
                "report_type": document_metadata[document_id]["report_type"],
                "release_year": document_metadata[document_id]["release_year"],
                "full_report": document_metadata[document_id]["full_report"],
                "page_label": node.metadata.get("label", "Unknown")
            }
        else:
            node.metadata = {
                "document_name": "Unknown Document",
                "report_type": "Unknown",
                "release_year": "Unknown",
                "full_report": "Unknown",
                "page_label": node.metadata.get("label", "Unknown") 
            }

    def process_node(node):
        node.embedding = create_embedding(node.text)
        return node

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_node, node): node for node in nodes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding Progress"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing node: {e}")

    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    return retriever


def retrieve_evidence(retriever, claim_text):
    relevant_nodes = retriever.retrieve(str_or_query_bundle=claim_text)
    evidence_list = []

    for node in relevant_nodes:
        evidence_list.append({
            "document_name": node.metadata.get("document_name"),
            "report_type": node.metadata.get("report_type"),
            "release_year": node.metadata.get("release_year"),
            "full_report": node.metadata.get("full_report"),
            "page_number": node.metadata.get("page_label"),
            "text": node.text
        })

    return evidence_list


def build_prompt(context, claim, supplement, evidence_list):
    evidence_strings = []
    for evidence in evidence_list:
        evidence_strings.append(
            f"- Document: {evidence['document_name']} ({evidence['report_type']}, {evidence['release_year']})\n"
            f"  Full Report: {evidence['full_report']}\n"
            f"  Page: {evidence['page_number']}.\n"
            f"  Text: {evidence['text']}\n"
        )

    evidence_section = "\n".join(evidence_strings)
    return f"""
You are a fact-checking agent. Your task is to determine the accuracy of the given claim and provide appropriate reasoning and sources.

### Role
1. Your role is to verify facts based on objective and reliable information.
2. When presenting evidence, you must include clear and accurate citations.
3. Always adhere to the response format below.

### Response Format
- This claim is (inaccurate/accurate/incorrect/misleading/not enough evidence).
- Because .....
- This can be verified by **(Document Name)** on **(Page).page**, where it states: **(Original Text)**.
- Additional Evidence:\n{evidence_section}

### Given Claim
"{claim}"

### Provided Context
"{context}"

### Additional Supplementary Information
"{supplement}"
"""


def demo_agent(context, claim, supplement, evidence_list):
    prompt = build_prompt(context, claim, supplement, evidence_list)

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
                    {"role": "system", "content": prompt}
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
    info_file = "info.json"
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
        TOP_K=5,
        info_file=info_file
    )
    evidence_list = retrieve_evidence(retriever, claim_text)
    context = "\n".join(evidence["text"] for evidence in evidence_list)

    result = demo_agent(context, claim_text, supplement, evidence_list)
    print(f"Claim: {claim_text}\n")
    print(f"Fact-check Result: {result}")


if __name__ == "__main__":
    main()
