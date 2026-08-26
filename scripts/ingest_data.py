import json
import os
import shutil

# NOTE: datainj/ no longer exists in this repo; this script is a historical one-time
# ingestion step, not part of the current data workflow. New synthetic data is authored
# directly into data/train, data/test, data/raw/rubrics.json — see
# scripts/generate_semantic_variants.py, scripts/generate_variants.py, and
# scripts/build_dataset_manifest.py instead.

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def ingest_file(input_file, output_dir):
    with open(input_file, "r") as f:
        data = json.load(f)
        
    for idx, record in enumerate(data):
        # Swap ai_suggested_scores to human_scores
        if "ai_suggested_scores" in record:
            record["human_scores"] = record.pop("ai_suggested_scores")
            
        record["human_reviewed"] = True
        
        # Remove placeholder if it exists (it probably doesn't, but safe to strip)
        if "placeholder" in record:
            del record["placeholder"]
            
        ans_id = record.get("answer_id", f"unknown_{idx}")
        out_path = os.path.join(output_dir, f"{ans_id}.json")
        
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

def main():
    print("Ingesting data from datainj...")
    clear_directory("data/train")
    clear_directory("data/test")
    
    ingest_file("datainj/train_answers_DRAFT.json", "data/train")
    ingest_file("datainj/test_answers_DRAFT.json", "data/test")
    
    shutil.copy("datainj/rubrics.json", "data/raw/rubrics.json")
    shutil.copy("datainj/manifest_DRAFT.json", "data/dataset_manifest.json")
    print("Data ingestion complete.")

if __name__ == "__main__":
    main()
