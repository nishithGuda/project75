"""
Dataset Integrator
Integrates data from multiple Rico datasets for the LLM-based UI Navigation Assistant.
"""

import os
import json
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetIntegrator:
    """
    Integrates data from the Rico datasets:
    1. Base Rico (screenshots and view hierarchies)
    2. Rico Semantic Annotations
    3. Rico RefExp (referential expressions)
    """
    
    def __init__(self, output_dir: str = "./processed_data"):
        """
        Initialize the integrator.
        
        Args:
            output_dir (str): Directory for processed output
        """
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "processed"
        self.project_dir = self.output_dir / "project_data"
        self.training_dir = self.project_dir / "training"
        
        # Create directories
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.project_dir.mkdir(exist_ok=True, parents=True)
        self.training_dir.mkdir(exist_ok=True, parents=True)
    
    def integrate(self, 
                 metadata: Optional[Dict] = None, 
                 semantics: Optional[Dict] = None, 
                 refexp: Optional[Dict] = None) -> Dict:
        """
        Integrate data from multiple datasets.
        
        Args:
            metadata (dict, optional): Processed metadata
            semantics (dict, optional): Processed semantic annotations
            refexp (dict, optional): Processed RefExp data
            
        Returns:
            dict: Integrated dataset
        """
        logger.info("Integrating datasets...")
        
        # Initialize integrated data
        integrated_data = {}
        
        # Load metadata if path provided instead of data
        if isinstance(metadata, str) and os.path.exists(metadata):
            logger.info(f"Loading metadata from {metadata}")
            with open(metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Load semantics if path provided instead of data
        if isinstance(semantics, str) and os.path.exists(semantics):
            logger.info(f"Loading semantics from {semantics}")
            with open(semantics, 'r', encoding='utf-8') as f:
                semantics = json.load(f)
        
        # Load refexp if path provided instead of data
        if isinstance(refexp, str) and os.path.exists(refexp):
            logger.info(f"Loading refexp from {refexp}")
            with open(refexp, 'r', encoding='utf-8') as f:
                refexp = json.load(f)
        
        # Add base metadata
        if metadata:
            logger.info(f"Adding metadata for {len(metadata)} screens")
            
            for screen_id, screen_data in metadata.items():
                integrated_data[screen_id] = {
                    "screen_id": screen_id,
                    "elements": screen_data.get("elements", []),
                    "screenshot_path": screen_data.get("screenshot_path", ""),
                    "hierarchy_path": screen_data.get("hierarchy_path", ""),
                    "sources": ["rico_base"]
                }
        
        # Add semantic annotations
        if semantics:
            logger.info(f"Adding semantic annotations for {len(semantics)} screens")
            
            for screen_id, semantic_data in semantics.items():
                if screen_id in integrated_data:
                    # Add to existing screen data
                    if "semantic_annotations" not in integrated_data[screen_id]:
                        integrated_data[screen_id]["semantic_annotations"] = []
                    
                    integrated_data[screen_id]["semantic_annotations"].extend(
                        semantic_data.get("annotations", [])
                    )
                    
                    if "rico_semantics" not in integrated_data[screen_id]["sources"]:
                        integrated_data[screen_id]["sources"].append("rico_semantics")
                else:
                    # Create new entry
                    integrated_data[screen_id] = {
                        "screen_id": screen_id,
                        "semantic_annotations": semantic_data.get("annotations", []),
                        "sources": ["rico_semantics"]
                    }
        
        # Add RefExp data
        if refexp:
            logger.info(f"Adding RefExp data for {len(refexp)} screens")
            
            for screen_id, refexp_data in refexp.items():
                if screen_id in integrated_data:
                    # Add to existing screen data
                    integrated_data[screen_id]["refexp_data"] = refexp_data.get("refexp_data", [])
                    
                    if "rico_refexp" not in integrated_data[screen_id]["sources"]:
                        integrated_data[screen_id]["sources"].append("rico_refexp")
                else:
                    # Create new entry
                    integrated_data[screen_id] = {
                        "screen_id": screen_id,
                        "refexp_data": refexp_data.get("refexp_data", []),
                        "sources": ["rico_refexp"]
                    }
        
        # Save integrated data
        output_path = self.processed_dir / "integrated_dataset.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f)
        
        logger.info(f"Saved integrated dataset with {len(integrated_data)} screens to {output_path}")
        
        return integrated_data
    
    def generate_project_data(self, integrated_data: Dict):
        """
        Generate data files for the project structure.
        
        Args:
            integrated_data (dict): Integrated dataset
        """
        logger.info("Generating project data files...")
        
        # Create example_metadata.json
        self._create_example_metadata(integrated_data)
        
        # Create history.json
        self._create_history_file(integrated_data)
        
        # Create training datasets
        self._create_training_datasets(integrated_data)
        
        logger.info("Project data generation complete")
    
    def copy_to_project(self, project_root: str = "./backend"):
        """
        Copy generated files to the project structure.
        
        Args:
            project_root (str): Root directory of the project
        """
        project_root = Path(project_root)
        data_dir = project_root / "data"
        
        # Create directories if they don't exist
        data_dir.mkdir(exist_ok=True, parents=True)
        (data_dir / "training").mkdir(exist_ok=True, parents=True)
        
        # Copy example_metadata.json
        if (self.project_dir / "example_metadata.json").exists():
            shutil.copy(
                self.project_dir / "example_metadata.json",
                data_dir / "example_metadata.json"
            )
        
        # Copy history.json
        if (self.project_dir / "history.json").exists():
            shutil.copy(
                self.project_dir / "history.json",
                data_dir / "history.json"
            )
        
        # Copy training files
        for training_file in self.training_dir.glob("*.json"):
            shutil.copy(
                training_file,
                data_dir / "training" / training_file.name
            )
        
        logger.info(f"Copied project data files to {data_dir}")
    
    def _create_example_metadata(self, integrated_data: Dict):
        """
        Create example_metadata.json file.
        
        Args:
            integrated_data (dict): Integrated dataset
        """
        # Select screens with data from multiple sources if possible
        multi_source_screens = [
            screen_id for screen_id, data in integrated_data.items()
            if len(data.get("sources", [])) >= 2
        ]
        
        # If not enough multi-source screens, include single-source screens
        if len(multi_source_screens) < 10:
            # Add screens with clickable elements
            screens_with_clickable = []
            for screen_id, data in integrated_data.items():
                if "elements" in data:
                    clickable_elements = [e for e in data["elements"] if e.get("clickable", False)]
                    if clickable_elements:
                        screens_with_clickable.append(screen_id)
            
            # Combine screens and take up to 10
            candidate_screens = list(set(multi_source_screens + screens_with_clickable))
            selected_screens = candidate_screens[:10]
            
            # If still not enough, add any screens
            if len(selected_screens) < 10:
                all_screens = list(integrated_data.keys())
                remaining = 10 - len(selected_screens)
                additional_screens = [s for s in all_screens if s not in selected_screens][:remaining]
                selected_screens.extend(additional_screens)
        else:
            selected_screens = multi_source_screens[:10]
        
        # Create example metadata list
        example_metadata = []
        for screen_id in selected_screens:
            if screen_id in integrated_data:
                example_metadata.append(integrated_data[screen_id])
        
        # Save to file
        output_path = self.project_dir / "example_metadata.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(example_metadata, f, indent=2)
        
        logger.info(f"Created example_metadata.json with {len(example_metadata)} screens")
    
    def _create_history_file(self, integrated_data: Dict):
        """
        Create history.json file with example interactions.
        
        Args:
            integrated_data (dict): Integrated dataset
        """
        history = []
        
        # Find screens with RefExp data
        refexp_screens = [
            screen_id for screen_id, data in integrated_data.items()
            if "refexp_data" in data and data["refexp_data"]
        ]
        
        # Create entries from RefExp data
        for screen_id in refexp_screens[:5]:  # Limit to 5 screens
            screen_data = integrated_data[screen_id]
            
            if "refexp_data" in screen_data and screen_data["refexp_data"]:
                # Use up to 2 RefExp examples per screen
                for refexp_item in screen_data["refexp_data"][:2]:
                    expression = refexp_item.get("expression", "")
                    if not expression:
                        continue
                    
                    # Create history entry
                    entry = {
                        "query": expression,
                        "screen_id": screen_id,
                        "action": {
                            "type": "click",
                            "target_idx": refexp_item.get("target_idx", 0),
                            "confidence": 0.9
                        },
                        "feedback": "success"
                    }
                    
                    # Add target bounding box if available
                    if "target_bbox" in refexp_item:
                        entry["action"]["target_bbox"] = refexp_item["target_bbox"]
                    
                    history.append(entry)
        
        # Find screens with clickable elements if RefExp data is insufficient
        if len(history) < 5:
            # Find screens with clickable elements
            screens_with_clickable = []
            for screen_id, data in integrated_data.items():
                if "elements" in data:
                    clickable_elements = [e for e in data["elements"] if e.get("clickable", False)]
                    if clickable_elements and screen_id not in refexp_screens[:5]:
                        screens_with_clickable.append((screen_id, clickable_elements))
            
            # Create history entries from clickable elements
            for screen_id, clickable_elements in screens_with_clickable[:5]:
                # Take up to 2 clickable elements per screen
                for i, element in enumerate(clickable_elements[:2]):
                    # Create a query based on element properties
                    query = ""
                    if element.get("text"):
                        query = f"Click on {element['text']}"
                    elif element.get("content_desc"):
                        query = f"Click on {element['content_desc']}"
                    elif element.get("type") != "unknown":
                        query = f"Click the {element['type']}"
                    else:
                        query = f"Click the element at position {element['bounds'][0]}, {element['bounds'][1]}"
                    
                    # Create history entry
                    entry = {
                        "query": query,
                        "screen_id": screen_id,
                        "action": {
                            "type": "click",
                            "target_idx": i,
                            "confidence": 0.85
                        },
                        "feedback": "success"
                    }
                    
                    history.append(entry)
                    
                    # Break if we have enough entries
                    if len(history) >= 5:
                        break
                
                # Break if we have enough entries
                if len(history) >= 5:
                    break
        
        # Add generic examples if we still don't have enough
        if len(history) < 5:
            generic_examples = [
                {
                    "query": "Go back to the main screen",
                    "action": {
                        "type": "click",
                        "target_type": "Button",
                        "target_desc": "Back",
                        "confidence": 0.95
                    },
                    "feedback": "success"
                },
                {
                    "query": "Click on the settings icon",
                    "action": {
                        "type": "click",
                        "target_type": "ImageView",
                        "target_desc": "Settings",
                        "confidence": 0.87
                    },
                    "feedback": "success"
                },
                {
                    "query": "Scroll down to see more items",
                    "action": {
                        "type": "scroll",
                        "direction": "down",
                        "confidence": 0.92
                    },
                    "feedback": "success"
                },
                {
                    "query": "Search for restaurants nearby",
                    "action": {
                        "type": "input",
                        "target_type": "EditText",
                        "target_desc": "Search",
                        "input_text": "restaurants nearby",
                        "confidence": 0.89
                    },
                    "feedback": "success"
                }
            ]
            
            # Add enough generic examples to have at least 5 total
            for example in generic_examples[:max(0, 5 - len(history))]:
                history.append(example)
        
        # Save to file
        output_path = self.project_dir / "history.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Created history.json with {len(history)} interactions")
    
    def _create_training_datasets(self, integrated_data: Dict):
        """
        Create training datasets for LLM engine and RL module.
        
        Args:
            integrated_data (dict): Integrated dataset
        """
        # Create LLM training data
        llm_training_data = []
        
        # Find screens with RefExp data
        refexp_screens = [
            screen_id for screen_id, data in integrated_data.items()
            if "refexp_data" in data and data["refexp_data"]
        ]
        
        # If no RefExp data, use screens with clickable elements
        if not refexp_screens:
            refexp_screens = [
                screen_id for screen_id, data in integrated_data.items()
                if "elements" in data and any(e.get("clickable", False) for e in data["elements"])
            ]
        
        # Create LLM training examples
        for screen_id in refexp_screens:
            screen_data = integrated_data[screen_id]
            
            # Get all interactive elements
            elements = []
            
            # Add elements from base data
            if "elements" in screen_data:
                elements.extend([
                    elem for elem in screen_data["elements"]
                    if elem.get("clickable", False)
                ])
            
            # Add elements from semantic annotations if available
            if "semantic_annotations" in screen_data:
                for i, annotation in enumerate(screen_data["semantic_annotations"]):
                    if "bbox" in annotation:
                        semantic_element = {
                            "id": f"semantic_{i}",
                            "type": annotation.get("component_type", annotation.get("type", "semantic")),
                            "class": annotation.get("class", ""),
                            "bounds": annotation["bbox"],
                            "clickable": annotation.get("clickable", True),
                            "from_semantic": True
                        }
                        elements.append(semantic_element)
            
            # Skip if no elements available
            if not elements:
                continue
            
            # Create examples from RefExp data if available
            if "refexp_data" in screen_data and screen_data["refexp_data"]:
                for refexp_item in screen_data["refexp_data"]:
                    expression = refexp_item.get("expression", "")
                    if not expression:
                        continue
                    
                    target_idx = refexp_item.get("target_idx", -1)
                    target_bbox = refexp_item.get("target_bbox", None)
                    
                    # Create LLM training example
                    example = {
                        "query": expression,
                        "screen_id": screen_id,
                        "elements": elements,
                        "target_idx": target_idx,
                        "split": refexp_item.get("split", "train")
                    }
                    
                    if target_bbox:
                        example["target_bbox"] = target_bbox
                    
                    llm_training_data.append(example)
            else:
                # Create examples based on element properties
                for i, element in enumerate(elements):
                    query = ""
                    if element.get("text"):
                        query = f"Click on {element['text']}"
                    elif element.get("content_desc"):
                        query = f"Click on {element['content_desc']}"
                    elif element.get("type") != "unknown":
                        query = f"Click the {element['type']}"
                    else:
                        continue  # Skip elements without good descriptions
                    
                    # Create LLM training example
                    example = {
                        "query": query,
                        "screen_id": screen_id,
                        "elements": elements,
                        "target_idx": i,
                        "split": "train"
                    }
                    
                    llm_training_data.append(example)
        
        # Create RL training data
        rl_training_data = []
        
        # For each LLM example, create an RL example
        for llm_example in llm_training_data:
            # Generate a simulated confidence score
            confidence = np.random.uniform(0.7, 1.0)
            
            rl_example = {
                "state": {
                    "query": llm_example["query"],
                    "screen_id": llm_example["screen_id"],
                    "confidence": confidence
                },
                "action": "click",  # Assuming click action for now
                "reward": 1.0,  # Assuming success for training data
                "next_state": None  # To be filled during training
            }
            
            rl_training_data.append(rl_example)
        
        # Save training datasets
        llm_output_path = self.training_dir / "llm_training_data.json"
        with open(llm_output_path, 'w', encoding='utf-8') as f:
            json.dump(llm_training_data, f, indent=2)
        
        rl_output_path = self.training_dir / "rl_training_data.json"
        with open(rl_output_path, 'w', encoding='utf-8') as f:
            json.dump(rl_training_data, f, indent=2)
        
        logger.info(f"Created LLM training dataset with {len(llm_training_data)} examples")
        logger.info(f"Created RL training dataset with {len(rl_training_data)} examples")


# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = DatasetIntegrator(output_dir="./processed_data")
    
    # Paths to processed data
    metadata_path = "./processed_data/processed/base/processed_metadata.json"
    semantics_path = "./processed_data/processed/semantics/processed_semantics.json"
    
    # Integrate datasets
    integrated_data = integrator.integrate(
        metadata=metadata_path,
        semantics=semantics_path
    )
    
    # Generate project data
    integrator.generate_project_data(integrated_data)
    
    # Copy to project
    integrator.copy_to_project(project_root="./backend")