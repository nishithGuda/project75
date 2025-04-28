from datasets import load_dataset
import time
import json
import os

start_time = time.time()
print("Starting dataset load...")

# Load screen IDs from processed_metadata.json
metadata_path = os.path.join("backend2", "datasets", "processed", "base", "processed_metadata.json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)
valid_screen_ids = set(metadata.keys())
print(f"✅ Loaded {len(valid_screen_ids)} valid screen_ids from metadata")

# Load the dataset with streaming
ds_streaming = load_dataset("ivelin/rico_refexp_combined", split="train", streaming=True)

collected_samples = []
sample_count = 0
max_samples_to_check = 5000
max_samples_to_collect = 1000

print("Starting filtering process...")
for sample in ds_streaming:
    sample_count += 1

    image_id = str(sample.get("image_id", ""))
    prompt = sample.get("prompt", "")
    bbox = sample.get("target_bounding_box", {})

    #Check required fields and metadata presence
    if not image_id or not prompt or not bbox:
        continue
    if image_id not in valid_screen_ids:
        continue

    collected_samples.append({
        "image_id": image_id,
        "prompt": prompt,
        "target_bounding_box": bbox
    })

    if len(collected_samples) % 100 == 0:
        print(f"Found {len(collected_samples)} valid samples out of {sample_count} checked...")

    if len(collected_samples) >= max_samples_to_collect or sample_count >= max_samples_to_check:
        break

elapsed_time = time.time() - start_time
print(f"✅ Collected {len(collected_samples)} valid samples in {elapsed_time:.2f} seconds.")
print(f"🔍 Checked {sample_count} total samples.")
print(collected_samples[0])

# Save to backend2/datasets
output_path = os.path.join("backend2", "datasets", "refexp_samples.json")
with open(output_path, "w") as f:
    json.dump(collected_samples, f, indent=2)

print(f"Saved filtered RefExp samples to {output_path}")
