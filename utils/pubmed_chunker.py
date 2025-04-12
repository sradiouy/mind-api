import os
import time
import pandas as pd
from Bio import Entrez
from bs4 import BeautifulSoup
import openai
import re
import json

# === CONFIG ===
# Friday, April 11, 2025 at 7:03:19 PM CEST
Entrez.email = "your@email.com"  # Replace with your email
openai.api_key = "" # Example key - Replace or use environment variables

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CHARS_PER_EMBED = 8000
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSIONS = 3072
OUTPUT_JSON = "pinecone_data_final_modules.json"
OUTPUT_CSV = "pinecone_data_final_modules.csv"

# === TOY DATASET METADATA LOOKUP ===
# IMPORTANT: Ensure the 'category' values below match the new ALLOWED_MODULES keys (e.g., use "literature" instead of "literature_review")
TOY_METADATA_LOOKUP = {
    # Ensure 'category' values match keys in ALLOWED_MODULES below
    "25036628": {"metadata": {"category": "design", "study_type": ["human", "animal"], "focus": ["study_design", "sampling_strategy", "metadata_standardization"], "keywords": ["reproducibility", "confounding_factors", "study_controls"]}},
    "28903732": {"metadata": {"category": "workflow", "sample_type": "chicken_cecum", "technologies": ["Illumina_MiSeq", "Ion_Torrent_PGM", "Roche_454"], "pipelines": ["QIIME", "UPARSE", "DADA2"], "keywords": ["platform_comparison", "bioinformatics", "sequencing_bias"]}}, # Changed category example
    "29158405": {"metadata": {"category": "literature", "study_type": "meta_analysis", "focus": ["disease_association", "microbiome_patterns"], "keywords": ["gut_microbiome", "disease_specific", "shared_responses"]}}, # Changed category example
    "30691412": {"metadata": {"category": "pipeline", "tools_supported": ["Mothur", "QIIME2"], "features": ["metadata_profiling", "quality_control", "diversity_analysis"], "keywords": ["pipeline_integration", "data_visualization", "user_friendly"]}}, # Changed category example
    "31604899": {"metadata": {"category": "protocol", "methods": ["16S_rRNA_sequencing", "shotgun_metagenomics"], "focus": ["protocol_comparison", "standardization", "best_practices"], "keywords": ["microbiome_analysis", "sequencing_protocols", "data_standardization"]}} # Changed category example
    # Add an example for 'trial' and 'learning' if you have corresponding PMIDs
    # "some_trial_pmid": {"metadata": {"category": "trial", ...}}
    # "some_learning_pmid": {"metadata": {"category": "learning", ...}}
}


# === STANDARDS & CONTROLLED VOCABULARIES (UPDATED) ===
# User-defined categories for the 'module' field
ALLOWED_MODULES = {
    "trial", "literature", "protocol", "workflow",
    "pipeline", "design", "learning",
    "unknown" # Fallback for unmapped categories
}
# Difficulty and Audience standards REMOVED

# Default values for fields that always exist or have a fallback
DEFAULT_MODULE = "unknown"
DEFAULT_TOPIC = "general_topic" # Topic can be refined based on module later if needed
DEFAULT_TAGS = []

# === NORMALIZATION HELPER FUNCTIONS (Unchanged) ===
def normalize_string(value, allowed_values=None, default_value="unknown"):
    """Normalizes a string: lowercase, strip whitespace. Validates against allowed_values if provided."""
    if not isinstance(value, str): value = str(value)
    normalized = value.lower().strip()
    if allowed_values:
        if normalized in allowed_values: return normalized
        else:
            normalized_alt = normalized.replace(" ", "_").replace("-", "_")
            if normalized_alt in allowed_values: return normalized_alt
            return default_value
    else:
        return normalized

def normalize_tags(tags_list):
    """Normalizes a list of tags: converts all to strings, normalizes each, removes duplicates and empty strings."""
    if not isinstance(tags_list, list):
        if isinstance(tags_list, str): tags_list = [tags_list]
        else: return []
    normalized_tags = set()
    for tag in tags_list:
        if tag is None: continue
        normalized = normalize_string(str(tag))
        if normalized: normalized_tags.add(normalized)
    return sorted(list(normalized_tags))

# === METADATA PROCESSING FUNCTION (REVISED - Removed Difficulty/Audience/Curated) ===
def get_metadata_for_pmid(pmid, pmcid):
    """
    Gets and standardizes metadata for a PMID using the new category list.
    Uses lookup if available. Fields 'difficulty', 'audience', 'curated' are EXCLUDED.
    """
    # --- Initialize fields that always exist or have defaults ---
    module = DEFAULT_MODULE
    topic = DEFAULT_TOPIC # Topic might be derived from module later
    tags = list(DEFAULT_TAGS)
    source = f"PMC:{pmcid}" if pmcid else f"PMID:{pmid}"

    # --- Check Lookup Table ---
    raw_metadata = {}
    if pmid in TOY_METADATA_LOOKUP:
        raw_metadata = TOY_METADATA_LOOKUP[pmid].get("metadata", {})

    # --- Base fields (Module, Topic, Tags) ---
    raw_category = raw_metadata.get("category")
    if raw_category:
        # Validate against the NEW ALLOWED_MODULES list
        module = normalize_string(raw_category, allowed_values=ALLOWED_MODULES, default_value=DEFAULT_MODULE)
        # Simple topic assignment based on module (can be refined)
        topic = module if module != "unknown" else DEFAULT_TOPIC

    # Aggregate tags from various potential fields in the lookup
    potential_tags = []
    fields_for_tags = ["keywords", "study_type", "focus", "sample_type",
                       "technologies", "pipelines", "tools_supported",
                       "features", "methods"]
    for field in fields_for_tags:
        value = raw_metadata.get(field)
        if isinstance(value, list): potential_tags.extend(value)
        elif isinstance(value, str) and value: potential_tags.append(value)
    tags = normalize_tags(potential_tags)

    # --- Assemble final metadata (EXCLUDING difficulty, audience, curated) ---
    standardized_metadata = {
        "module": module,
        "source": source,
        "topic": topic, # Topic is derived from module or default
        "tags": tags,
    }

    return standardized_metadata


# === CORE FUNCTIONS (fetch_pmcid_from_pmid, fetch_full_text_pmc, chunk_text, embed_text - unchanged) ===
def fetch_pmcid_from_pmid(pmid):
    try:
        with Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc_refs") as handle:
            record = Entrez.read(handle)
        if record and record[0].get("LinkSetDb") and record[0]["LinkSetDb"][0].get("Link"):
            return record[0]["LinkSetDb"][0]["Link"][0]["Id"]
        else: return None
    except Exception as e: return None

def fetch_full_text_pmc(pmcid):
    try:
        with Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml") as handle:
            xml = handle.read()
        soup = BeautifulSoup(xml, "lxml-xml")
        body = soup.find("body")
        if body:
            text_parts = [p.get_text() for p in body.find_all(['p', 'sec'])]
            full_text = "\n".join(text_parts); full_text = re.sub(r'\s+', ' ', full_text).strip()
            return full_text
        else: return None
    except Exception as e: return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []; start = 0; text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == text_len: break
        start += chunk_size - overlap
        if start >= end: start = end
    return chunks

def embed_text(text, model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS):
    if not text: return None
    try:
        if len(text) > MAX_CHARS_PER_EMBED: text = text[:MAX_CHARS_PER_EMBED]
        response = openai.embeddings.create(input=[text], model=model, dimensions=dimensions)
        return response.data[0].embedding
    except openai.RateLimitError:
        print("Rate limit exceeded, sleeping..."); time.sleep(60); return embed_text(text, model, dimensions)
    except Exception as e: print(f"Embedding failed: {e}"); return None


# === MAIN WORKFLOW (Unchanged logic, uses revised get_metadata_for_pmid) ===
def process_pmids(pmids):
    all_data_for_pinecone = []
    for i, pmid in enumerate(pmids):
        print(f"\n🔍 Processing PMID {pmid} ({i+1}/{len(pmids)})")
        pmcid = fetch_pmcid_from_pmid(pmid)
        if not pmcid: print(f" PMID {pmid}: No PMCID. ", end="")

        # --- Get Standardized Metadata (Now excludes difficulty/audience/curated) ---
        base_metadata = get_metadata_for_pmid(pmid, pmcid)

        if not pmcid: print("Skipping text processing."); continue

        print(f" PMCID {pmcid}: Fetching text... ", end="")
        text = fetch_full_text_pmc(pmcid)
        if not text: print("Failed. Skipping text processing."); continue
        print(f"{len(text)} chars. Chunking... ", end="")

        chunks = chunk_text(text)
        print(f"{len(chunks)} chunks. Embedding... ", end="")

        embedded_count = 0
        for j, (chunk_text_content, start, end) in enumerate(chunks):
            embedding = embed_text(chunk_text_content)
            if not embedding: continue
            embedded_count += 1

            pinecone_id = f"{base_metadata.get('module', DEFAULT_MODULE)}_{pmcid}_{j}"

            chunk_data = {
                "id": pinecone_id, "values": embedding,
                "metadata": { **base_metadata, # Merges the new base metadata
                              "text": chunk_text_content, "pmid": str(pmid),
                              "pmcid": str(pmcid), "chunk_id": j,
                              "char_start": start, "char_end": end, }
            }
            all_data_for_pinecone.append(chunk_data)

        print(f"Embedded {embedded_count}/{len(chunks)} chunks.")
        time.sleep(0.4)

    return all_data_for_pinecone

# === SAVE FUNCTION ===
def save_data(data, filename):
    try:
        with open(filename, 'w') as f: json.dump(data, f, indent=4)
        print(f"✅ Standardized data saved to {filename}")
    except Exception as e: print(f"Error saving data to {filename}: {e}")

# === RUN SCRIPT ===
if __name__ == "__main__":
    # Ensure the PMIDs here correspond to categories in your updated TOY_METADATA_LOOKUP
    pmids_to_process = list(TOY_METADATA_LOOKUP.keys())
    pmids_to_process.extend(["12345678", "98765432"]) # Example unknown PMIDs

    print("Starting standardization and processing script with updated modules...")
    structured_data = process_pmids(pmids_to_process)

    if structured_data:
        save_data(structured_data, OUTPUT_JSON)
        # Optional: Save flattened CSV
        try:
            import pandas as pd
            # Define the exact columns expected based on the final metadata structure
            expected_metadata_keys = ['module', 'source', 'topic', 'tags',
                                     'text', 'pmid', 'pmcid', 'chunk_id',
                                     'char_start', 'char_end']
            flat_data = []
            for item in structured_data:
                 meta = item.get('metadata', {})
                 flat_item = {'pinecone_id': item.get('id'),
                              'embedding_vector': "|".join(map(str, item.get('values', [])))}
                 # Add metadata fields, handling tags specifically
                 for key in expected_metadata_keys:
                     if key == 'tags':
                         flat_item[key] = ",".join(meta.get(key, []))
                     else:
                         flat_item[key] = meta.get(key) # Get value or None if somehow missing (shouldn't be)
                 flat_data.append(flat_item)

            df = pd.DataFrame(flat_data)
            # Define column order for CSV
            column_order = ['pinecone_id', 'embedding_vector'] + expected_metadata_keys
            df = df[column_order] # Reorder

            df.to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Flattened standardized data saved to {OUTPUT_CSV}")
        except ImportError: print("Note: pandas library not found, skipping CSV output. Install with: pip install pandas")
        except Exception as e: print(f"Could not save flattened data to CSV: {e}")
    else:
        print("No data was generated.")

    print("Script finished.")