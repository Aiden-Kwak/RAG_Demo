import os
import json
import requests
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from openai import AuthenticationError, APIConnectionError, RateLimitError, OpenAIError
from llama_index.core.node_parser import SentenceSplitter
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
        return np.array(embedding, dtype=np.float32)
    except (AuthenticationError, APIConnectionError, RateLimitError, OpenAIError) as e:
        print(f"OpenAI API Error: {e}")
        exit(1)

def save_faiss_index(index, metadata, index_file="faiss_index.bin", metadata_file="metadata.json"):
    faiss.write_index(index, index_file)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    print(f"Faiss index saved to {index_file}, metadata saved to {metadata_file}.")

def load_faiss_index(index_file="faiss_index.bin", metadata_file="metadata.json"):
    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        print("Faiss index or metadata file not found. Generating new index.")
        return None, None
    index = faiss.read_index(index_file)
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    print(f"Faiss index loaded from {index_file}, metadata loaded from {metadata_file}.")
    return index, metadata

def createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, info_file, index_file="faiss_index.bin", metadata_file="metadata.json", force_update=False):
    if not force_update:
        index, metadata = load_faiss_index(index_file, metadata_file)
        if index and metadata:
            return index, metadata

    documents = SimpleDirectoryReader(input_files=[REPORT]).load_data()
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)

    document_metadata = load_document_metadata(info_file)
    document_id = os.path.basename(REPORT).split(".")[0]

    embeddings = []
    metadata = []

    def process_node(node):
        if document_id in document_metadata:
            node_metadata = {
                "document_name": document_metadata[document_id]["title"],
                "report_type": document_metadata[document_id]["report_type"],
                "release_year": document_metadata[document_id]["release_year"],
                "full_report": document_metadata[document_id]["full_report"],
                "page_label": node.metadata.get("label", "Unknown"),
                "text": node.text
            }
        else:
            node_metadata = {
                "document_name": "Unknown Document",
                "report_type": "Unknown",
                "release_year": "Unknown",
                "full_report": "Unknown",
                "page_label": node.metadata.get("label", "Unknown"),
                "text": node.text
            }

        embedding = create_embedding(node.text)
        return embedding, node_metadata

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_node, node): node for node in nodes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Nodes"):
            try:
                embedding, node_metadata = future.result()
                embeddings.append(embedding)
                metadata.append(node_metadata)
            except Exception as e:
                print(f"Error processing node: {e}")

    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    save_faiss_index(index, metadata, index_file, metadata_file)

    return index, metadata

def retrieve_evidence(index, metadata, claim_text, top_k=5):
    query_embedding = create_embedding(claim_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    evidence_list = []
    for i in indices[0]:
        evidence_list.append(metadata[i])

    return evidence_list

def build_prompt(context, claim, supplement, evidence_list):
    evidence_strings = []
    for evidence in evidence_list:
        evidence_strings.append(
            f"- Document: {evidence['document_name']} ({evidence['report_type']}, {evidence['release_year']})\n"
            f"  Full Report: {evidence['full_report']}\n"
            f"  Page: {evidence['page_label']}\n"
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

    index, metadata = createRetriever(
        REPORT=pdf_file,
        CHUNK_SIZE=512,
        CHUNK_OVERLAP=50,
        info_file=info_file,
        force_update=False
    )
    evidence_list = retrieve_evidence(index, metadata, claim_text, top_k=5)
    context = "\n".join(evidence["text"] for evidence in evidence_list)

    result = demo_agent(context, claim_text, supplement, evidence_list)
    print(f"Claim: {claim_text}\n")
    print(f"Fact-check Result: {result}")

if __name__ == "__main__":
    main()
