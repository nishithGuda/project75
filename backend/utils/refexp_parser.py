"""
Rico RefExp Processor
Processes the Rico RefExp dataset containing referential expressions for UI elements.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RicoRefExpProcessor:
    """
    Processor for the Rico RefExp dataset (referential expressions).
    """
    
    def __init__(self, output_dir: str = "./processed_data"):
        """
        Initialize the RefExp processor.
        
        Args:
            output_dir (str): Directory for processed output
        """
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "processed" / "refexp"
        
        # Create directory
        self.processed_dir.mkdir(exist_ok=True, parents=True)

    def process_from_huggingface(self, dataset_name: str = "ivelin/rico_refexp_combined") -> Dict:
       
        try:
            # Import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                logger.error("Hugging Face datasets library not found. Installing...")
                import subprocess
                subprocess.check_call(["pip", "install", "datasets"])
                from datasets import load_dataset
            
            logger.info(f"Loading dataset '{dataset_name}' from Hugging Face")
            
            # Load the dataset with streaming enabled for better memory efficiency
            try:
                dataset = load_dataset(dataset_name, streaming=True)
                logger.info(f"Successfully loaded dataset")
            except Exception as e:
                logger.error(f"Error loading dataset '{dataset_name}': {e}")
                logger.info("Trying alternative dataset 'ivelin/rico_sca_refexp_synthetic'")
                try:
                    dataset = load_dataset("ivelin/rico_sca_refexp_synthetic", streaming=True)
                    logger.info(f"Successfully loaded alternative dataset")
                except Exception as e2:
                    logger.error(f"Error loading alternative dataset: {e2}")
                    return {}
            
            # Process the dataset
            processed_data = {}
            
            # Process each split (train, validation, test)
            for split_name, split_data in dataset.items():
                logger.info(f"Processing {split_name} split")
                
                # Track unique screen IDs to limit processing
                seen_screens = set()
                sample_count = 0
                max_samples_to_check = 5000  # Prevent endless iteration
                
                # Stream through examples
                progress_bar = tqdm(desc=f"Processing {split_name}")
                
                for example in split_data:
                    sample_count += 1
                    progress_bar.update(1)
                    
                    try:
                        # Extract screen/image ID - use the actual field name from your sample
                        screen_id = example.get('image_id')
                        if not screen_id:
                            continue
                        
                        # Check if we've already seen this screen
                        if screen_id in seen_screens and len(seen_screens) >= 10:
                            continue
                            
                        seen_screens.add(screen_id)
                        
                        # Initialize data for this screen
                        if screen_id not in processed_data:
                            processed_data[screen_id] = {
                                "screen_id": screen_id,
                                "refexp_data": []
                            }
                        
                        # Extract referring expression (from 'prompt' field in your case)
                        refexp = example.get('prompt', '')
                        if not refexp:
                            continue
                        
                        # Get bounding box information - use actual structure from your sample
                        target_bbox = example.get('target_bounding_box')
                        
                        # Create RefExp entry
                        refexp_entry = {
                            "expression": refexp,
                            "split": split_name
                        }
                        
                        if target_bbox:
                            refexp_entry["target_bbox"] = target_bbox
                        
                        # Add to processed data
                        processed_data[screen_id]["refexp_data"].append(refexp_entry)
                        
                        # Break if we've found enough unique screens
                        if len(seen_screens) >= 10 and sample_count >= 100:
                            logger.info(f"Found {len(seen_screens)} unique screens after checking {sample_count} samples")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing example: {e}")
                    
                    # Safety check to avoid running forever
                    if sample_count >= max_samples_to_check:
                        logger.warning(f"Reached maximum sample check limit ({max_samples_to_check})")
                        break
                
                progress_bar.close()
                logger.info(f"Checked {sample_count} samples in {split_name}")
            
            # Save processed data
            output_path = self.processed_dir / "processed_refexp.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f)
            
            logger.info(f"Saved processed RefExp data to {output_path}")
            logger.info(f"Processed {len(processed_data)} screens with RefExp data")
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error processing RefExp dataset: {e}")
            return {}
    def _extract_screen_id(self, example: Dict) -> Optional[str]:
        """
        Extract screen ID from example.
        
        Args:
            example (dict): Example data
            
        Returns:
            str or None: Screen ID
        """
        # Try different field names for screen ID
        if "screen_id" in example:
            return example["screen_id"]
        elif "image_id" in example:
            return example["image_id"]
        
        # Try to extract from image path
        if "image_path" in example:
            image_path = example["image_path"]
            if isinstance(image_path, str) and image_path:
                return Path(image_path).stem
        
        return None