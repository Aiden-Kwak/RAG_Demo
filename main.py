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
from functools import lru_cache

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
openai_api_key = os.getenv("OPENAI_API_KEY")

from PyPDF2 import PdfReader

def extract_text_with_page_numbers(pdf_path):
    reader = PdfReader(pdf_path)
    nodes = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        nodes.append({
            "page_number": i + 1,  # Page numbers start from 1
            "text": text.strip()
        })
    return nodes


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

@lru_cache(maxsize=10000)
def create_embedding_with_cache(text):
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
        raise e

def process_node(node, document_metadata, document_id):
    embedding = create_embedding_with_cache(node["text"])
    node_metadata = {
        "page_label": node["page_label"],
        "text": node["text"],
        **document_metadata.get(document_id, {
            "document_name": "Unknown Document",
            "report_type": "Unknown",
            "release_year": "Unknown",
            "full_report": "Unknown"
        })
    }
    return embedding, node_metadata

def createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, info_file, index_file="faiss_index.bin", metadata_file="metadata.json", force_update=False):
    if not force_update:
        index, metadata = load_faiss_index(index_file, metadata_file)
        if index and metadata:
            return index, metadata

    # PDF 텍스트와 페이지 번호 추출
    pdf_nodes = extract_text_with_page_numbers(REPORT)

    # 텍스트를 CHUNK_SIZE로 분할
    nodes = []
    for node in pdf_nodes:
        text = node["text"]
        page_number = node["page_number"]
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            nodes.append({
                "page_label": page_number,
                "text": chunk_text
            })

    # 문서 메타데이터 로드
    document_metadata = load_document_metadata(info_file)
    document_id = os.path.basename(REPORT).split(".")[0]

    embeddings = []
    metadata = []

    # 병렬 작업으로 노드 처리
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_node, node, document_metadata, document_id): node
            for node in nodes
        }
        with tqdm(total=len(futures), desc="Processing Nodes") as pbar:
            for future in as_completed(futures):
                try:
                    embedding, node_metadata = future.result()
                    embeddings.append(embedding)
                    metadata.append(node_metadata)
                except Exception as e:
                    print(f"Error processing node: {e}")
                finally:
                    pbar.update(1)

    # FAISS 인덱스 생성
    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 인덱스와 메타데이터 저장
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
    """
    Constructs a fact-checking prompt with a clear structure, logical reasoning, and reliable evidence citations,
    explicitly providing source titles and page details for each evidence.

    Parameters:
        context (str): Background information related to the claim.
        claim (str): The specific claim to be fact-checked.
        supplement (str): Additional context or supplementary information to aid the fact-check.
        evidence_list (list of dict): A list of evidence dictionaries, each containing title, report type, release year,
                                      full report link, page label, and a relevant excerpt of text.

    Returns:
        str: A formatted prompt for a fact-checking task.
    """
    # Prepare formatted evidence strings
    evidence_strings = []
    for evidence in evidence_list:
        evidence_strings.append(
            f"- **Source Title:** {evidence['title']}\n"
            f"  **Report Type:** {evidence['report_type']} ({evidence['release_year']})\n"
            f"  **Page:** {evidence['page_label']}\n"
            f"  **Full Report Link:** {evidence['full_report']}\n"
            f"  **Excerpt:** {evidence['text']}\n"
        )
    evidence_section = "\n".join(evidence_strings)

    # Build the fact-checking prompt
    return f"""
You are an expert fact-checking agent tasked with evaluating the accuracy of claims using reliable, objective, and scientific evidence.

### Your Role
1. Evaluate the accuracy of the claim based on provided evidence and context.
2. Present your findings clearly and logically, ensuring your reasoning is accessible to both experts and the general audience.
3. Support your reasoning with verifiable and properly cited sources, explicitly referencing the source titles and page details.

### Response Format
- **Claim Evaluation:** This claim is (accurate/inaccurate/misleading/incorrect/not enough evidence).
- **Reasoning:** A concise explanation of why the claim is or is not valid, backed by scientific evidence.
- **Evidence:** Provide direct citations from trusted sources, explicitly naming the source title and page details, to validate your reasoning.
- **Conclusion:** Summarize your findings in a way that is understandable to both technical and non-technical readers.

### Claim to Evaluate
"{claim}"

### Contextual Information
"{context}"

### Additional Supplementary Information
"{supplement}"

### Evidence
{evidence_section}
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
