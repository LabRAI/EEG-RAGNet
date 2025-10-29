import pdfplumber
import json
import logging
import time
from tqdm import tqdm
from openai import OpenAI


# Global Settings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

api_key = "sk-proj-XXXXXX_your_key_here" #replace with your own openai API key
client = OpenAI(api_key=api_key)

# ----------------------------
# PDF to Text Extraction
# ----------------------------
def pdf_to_pages(pdf_path):
    """Extract text from PDF, one string per page"""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
            else:
                pages_text.append(f"[Page {i} has no extractable text]")
    return pages_text


def build_raw_knowledge():
    """Build raw text corpus from PDF files"""
    pdf_files = [
        "Japan_Neurology_guideline2018.pdf",
        "NICE_UK_Epilepsy_Standard_2025.pdf",
        "SIGN_Scotland_Epilepsy_Standard_2018.pdf",
        "AES_Epilepsy_Guidelines.pdf",
        "ILAE_Epilepsy_Guidelines.pdf"
    ]
    knowledge_base = []

    for pdf in pdf_files:
        pages = pdf_to_pages(pdf)
        knowledge_base.extend(pages)
        print(f"Processed {pdf}, extracted {len(pages)} pages")

    json_file = "knowledge.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    print(f"Generated {json_file}, total {len(knowledge_base)} pages of text")

    return json_file


# ----------------------------
# Utility Functions
# ----------------------------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_output": text}


def load_json_lines(path):
    """Load knowledge.json as a list of strings"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            docs.append(str(d))
    return docs


def throttled_sleep(page_idx, sleep_per_page=3, sleep_every_n=20, long_sleep=60):
    """Throttle to avoid API rate limits"""
    time.sleep(sleep_per_page)
    if (page_idx + 1) % sleep_every_n == 0:
        print(f"Reached {page_idx+1} pages, taking a long break ({long_sleep}s)...")
        time.sleep(long_sleep)


# ----------------------------
# Step 1: Entity Extraction (NER)
# ----------------------------
def extract_entities_with_gpt(text):
    """Extract biomedical entities using GPT"""
    prompt = f"""
You are a professional biomedical NER (Named Entity Recognition) system.

Given the following medical text, extract a comprehensive list of key biomedical entities.
Return your output as a strict JSON object with keys as entity types and values as arrays of strings.

Text:
\"\"\"{text}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a biomedical entity extraction model."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        output_text = response.choices[0].message.content
        entities = safe_json_parse(output_text)
        return entities
    except Exception as e:
        print(f"GPT extraction failed: {e}")
        return {}


# ----------------------------
# Step 2: Relation Extraction (RE)
# ----------------------------
def extract_relations_with_gpt(entities, text):
    """
    Input: Extracted entities + original text
    Output: Triplets [{"head": ..., "relation": ..., "tail": ...}]
    """
    prompt = f"""
You are a biomedical relation extraction system.

Given the following text and extracted biomedical entities, infer meaningful relationships among them.
Only output explicit, verifiable relations that are medically relevant, such as:
- Drug treats Disease
- Test diagnoses Disease
- Symptom indicates Disease
- RiskFactor causes Disease
- Surgery treats Condition

Return a JSON list of triplets with "head", "relation", and "tail" keys.
Do NOT include explanations or markdown.

Text:
\"\"\"{text}\"\"\"

Entities:
{json.dumps(entities, ensure_ascii=False)}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a biomedical relation extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=600,
        )
        output_text = response.choices[0].message.content
        triplets = safe_json_parse(output_text)

        # Format output
        if isinstance(triplets, list):
            clean_triplets = []
            for t in triplets:
                if isinstance(t, dict) and all(k in t for k in ["head", "relation", "tail"]):
                    clean_triplets.append(t)
            return clean_triplets
        else:
            return []
    except Exception as e:
        print(f"GPT relation extraction failed: {e}")
        return []


# ----------------------------
# Step 3: Processing Pipeline
# ----------------------------
def process_documents(docs, sleep_per_page=3, sleep_every_n=20, long_sleep=60):
    """Run NER and RE for each page"""
    results = []
    for idx, doc in enumerate(tqdm(docs, desc="Processing documents")):
        text = str(doc)
        entities = extract_entities_with_gpt(text)
        triplets = extract_relations_with_gpt(entities, text)

        results.append({
            "page_id": idx,
            "text": text,
            "entities": entities,
            "triplets": triplets
        })

        throttled_sleep(idx, sleep_per_page, sleep_every_n, long_sleep)
    return results


def save_to_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Step 1: Extracting raw text from PDFs ...")
    knowledge_file = build_raw_knowledge()

    print("Step 2: Performing entity & relation extraction ...")
    docs = load_json_lines(knowledge_file)
    ner_re_results = process_documents(docs, sleep_per_page=3, sleep_every_n=20, long_sleep=60)

    save_to_json(ner_re_results, "KG_triplets.json")
    print("Knowledge graph triplets saved to KG_triplets.json")
