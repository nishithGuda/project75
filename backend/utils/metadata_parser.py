"""
Rico Metadata Processor
Processes the base Rico dataset containing screenshots and view hierarchies.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RicoMetadataProcessor:
    """
    Processor for the base Rico dataset with screenshots and view hierarchies.
    """
    
    def __init__(self, output_dir: str = "./processed_data"):
        """
        Initialize the metadata processor.
        
        Args:
            output_dir (str): Directory for processed output
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw" / "base"
        self.processed_dir = self.output_dir / "processed" / "base"
        
        # Create directories
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
    
    def process(self, dataset_path: str, sample_size: int = 100) -> Dict:
        """
        Process the Rico base dataset.
        
        Args:
            dataset_path (str): Path to Rico base dataset
            sample_size (int): Number of screens to process
            
        Returns:
            dict: Processed metadata
        """
        dataset_path = Path(dataset_path)
        
        # Find screenshots and hierarchy files
        screenshots = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        hierarchies = list(dataset_path.glob("*.json"))
        
        if not screenshots:
            logger.warning("No screenshot files found in the dataset path")
        
        if not hierarchies:
            logger.warning("No hierarchy JSON files found in the dataset path")
        
        # Map screen IDs to files
        screen_map = {}
        
        # Map screenshots
        for screenshot in screenshots:
            screen_id = screenshot.stem
            if screen_id not in screen_map:
                screen_map[screen_id] = {"screenshot": str(screenshot)}
            else:
                screen_map[screen_id]["screenshot"] = str(screenshot)
        
        # Map hierarchies
        for hierarchy in hierarchies:
            screen_id = hierarchy.stem
            if screen_id not in screen_map:
                screen_map[screen_id] = {"hierarchy": str(hierarchy)}
            else:
                screen_map[screen_id]["hierarchy"] = str(hierarchy)
        
        # Only keep screens with both screenshot and hierarchy
        valid_screens = {
            k: v for k, v in screen_map.items() 
            if "screenshot" in v and "hierarchy" in v
        }
        
        logger.info(f"Found {len(valid_screens)} screens with both screenshot and hierarchy")
        
        # Limit to sample size
        screen_ids = list(valid_screens.keys())[:sample_size]
        
        # Process screens
        processed_data = {}
        
        for screen_id in tqdm(screen_ids):
            try:
                screen_data = valid_screens[screen_id]
                
                # Load hierarchy
                with open(screen_data["hierarchy"], 'r', encoding='utf-8') as f:
                    hierarchy_data = json.load(f)
                
                # Extract elements
                elements = self._extract_elements_from_rico(hierarchy_data)
                
                # Add to processed data
                processed_data[screen_id] = {
                    "screen_id": screen_id,
                    "screenshot_path": screen_data["screenshot"],
                    "hierarchy_path": screen_data["hierarchy"],
                    "elements": elements
                }
            except Exception as e:
                logger.error(f"Error processing screen {screen_id}: {e}")
        
        # Save processed data
        output_path = self.processed_dir / "processed_metadata.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f)
        
        logger.info(f"Saved processed metadata to {output_path}")
        
        return processed_data
    
    def _extract_elements_from_rico(self, hierarchy_data: Dict) -> List[Dict]:
        """
        Extract UI elements from a Rico view hierarchy.
        
        Args:
            hierarchy_data (dict): Rico view hierarchy data
            
        Returns:
            list: Extracted UI elements
        """
        elements = []
        
        # Handle different Rico JSON formats
        if "activity" in hierarchy_data and "root" in hierarchy_data["activity"]:
            # First format - with activity and root
            root = hierarchy_data["activity"]["root"]
            self._extract_elements_recursive_rico(root, elements)
        elif "root" in hierarchy_data:
            # Second format - with just root
            root = hierarchy_data["root"]
            self._extract_elements_recursive_rico(root, elements)
        elif "children" in hierarchy_data:
            # Third format - starting with the element directly
            self._extract_elements_recursive_rico(hierarchy_data, elements)
        else:
            # Try to find any structure that looks like a UI element
            logger.warning("Unknown Rico hierarchy structure, attempting to extract what we can")
            for key, value in hierarchy_data.items():
                if isinstance(value, dict) and "children" in value:
                    self._extract_elements_recursive_rico(value, elements)
        
        return elements
    
    def _extract_elements_recursive_rico(self, node: Dict, elements_list: List, parent_id: Optional[str] = None, depth: int = 0):
        """
        Recursively extract elements from a Rico hierarchy node.
        
        Args:
            node (dict): Node in the hierarchy
            elements_list (list): List to append elements to
            parent_id (str, optional): ID of the parent element
            depth (int): Current depth in the hierarchy
        """
        if not node or not isinstance(node, dict):
            return
        
        # Generate element ID
        element_id = f"elem_{len(elements_list)}"
        if "pointer" in node:
            element_id = f"elem_{node['pointer']}"
        
        # Extract bounds
        bounds = []
        if "bounds" in node and isinstance(node["bounds"], list) and len(node["bounds"]) == 4:
            bounds = node["bounds"]
        
        # Determine element class
        element_class = ""
        if "class" in node:
            element_class = node["class"]
        
        # Extract text content
        text = ""
        if "text" in node and node["text"]:
            if isinstance(node["text"], list) and len(node["text"]) > 0:
                text = node["text"][0] if node["text"][0] is not None else ""
            elif isinstance(node["text"], str):
                text = node["text"]
        
        # Extract content description
        content_desc = ""
        if "content-desc" in node and node["content-desc"]:
            if isinstance(node["content-desc"], list) and len(node["content-desc"]) > 0:
                content_desc = node["content-desc"][0] if node["content-desc"][0] is not None else ""
            elif isinstance(node["content-desc"], str):
                content_desc = node["content-desc"]
        
        # Determine if element is clickable
        clickable = False
        if "clickable" in node:
            clickable = node["clickable"]
        
        # Determine if element is enabled
        enabled = True
        if "enabled" in node:
            enabled = node["enabled"]
        
        # Determine if element is visible
        visible = True
        if "visibility" in node:
            visible = node["visibility"] == "visible"
        elif "visible-to-user" in node:
            visible = node["visible-to-user"]
        
        # Determine element type based on class name
        element_type = self._get_element_type_rico(element_class)
        
        # Create element
        element = {
            "id": element_id,
            "parent_id": parent_id,
            "class": element_class,
            "type": element_type,
            "text": text,
            "content_desc": content_desc,
            "clickable": clickable,
            "enabled": enabled,
            "visible": visible,
            "bounds": bounds,
            "depth": depth
        }
        
        # Add resource ID if available
        if "resource-id" in node:
            element["resource_id"] = node["resource-id"]
        
        # Add to list if it has meaningful bounds
        if bounds and (bounds[2] > bounds[0] or bounds[3] > bounds[1]):
            elements_list.append(element)
        
        # Process children
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                self._extract_elements_recursive_rico(child, elements_list, element_id, depth + 1)
    
    def _get_element_type_rico(self, class_name: str) -> str:
        """
        Determine element type from Rico class name.
        
        Args:
            class_name (str): Class name from Rico
            
        Returns:
            str: Element type
        """
        class_name = class_name.lower()
        
        # Map common Android class names to element types
        if "button" in class_name:
            return "button"
        elif "edittext" in class_name or "input" in class_name:
            return "input"
        elif "imageview" in class_name or "image" in class_name:
            return "image"
        elif "textview" in class_name or "text" in class_name:
            return "text"
        elif "checkbox" in class_name:
            return "checkbox"
        elif "radio" in class_name:
            return "radio"
        elif "spinner" in class_name or "dropdown" in class_name:
            return "dropdown"
        elif "switch" in class_name:
            return "switch"
        elif "layout" in class_name:
            return "layout"
        elif "webview" in class_name:
            return "webview"
        elif "view" in class_name:
            return "view"
        else:
            return "unknown"


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = RicoMetadataProcessor(output_dir="./processed_data")
    
    # Process dataset
    processed_data = processor.process(dataset_path="./data/combined", sample_size=10)
    
    print(f"Processed {len(processed_data)} screens")