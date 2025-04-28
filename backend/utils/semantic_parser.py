"""
Rico Semantic Annotations Processor
Processes the Rico Semantic Annotations dataset.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RicoSemanticProcessor:
    """
    Processor for the Rico Semantic Annotations dataset.
    """
    
    def __init__(self, output_dir: str = "./processed_data"):
        """
        Initialize the semantic processor.
        
        Args:
            output_dir (str): Directory for processed output
        """
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "processed" / "semantics"
        
        # Create directory
        self.processed_dir.mkdir(exist_ok=True, parents=True)
    
    def process(self, dataset_path: str) -> Dict:
        """
        Process the Rico Semantic Annotations dataset.
        
        Args:
            dataset_path (str): Path to Rico Semantic Annotations dataset
            
        Returns:
            dict: Processed semantic annotations
        """
        dataset_path = Path(dataset_path)
        
        # Find annotation files and images
        annotation_files = list(dataset_path.glob("**/*.json"))
        images = list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpg"))
        
        logger.info(f"Found {len(annotation_files)} annotation files and {len(images)} images")
        
        # Process annotation files
        processed_data = {}
        
        for annotation_file in tqdm(annotation_files):
            try:
                # Extract screen ID from filename
                screen_id = annotation_file.stem
                
                # Load annotation file
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                # Process semantic annotations for this screen
                annotations = self._process_rico_semantic_annotation(annotation_data, screen_id)
                
                # Add to processed data
                if annotations:
                    if screen_id not in processed_data:
                        processed_data[screen_id] = {
                            "screen_id": screen_id,
                            "annotations": annotations
                        }
                    else:
                        processed_data[screen_id]["annotations"].extend(annotations)
            except Exception as e:
                logger.error(f"Error processing annotation file {annotation_file}: {e}")
        
        # Save processed data
        output_path = self.processed_dir / "processed_semantics.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f)
        
        logger.info(f"Saved processed semantic annotations to {output_path}")
        
        return processed_data
    
    def _process_rico_semantic_annotation(self, annotation_data: Dict, screen_id: str) -> List[Dict]:
        """
        Process a Rico semantic annotation for a screen.
        
        Args:
            annotation_data (dict): Annotation data
            screen_id (str): Screen ID
            
        Returns:
            list: Processed annotations
        """
        annotations = []
        
        # Handle different annotation formats
        if "children" in annotation_data:
            # Format with children array
            self._process_semantic_node(annotation_data, annotations)
        elif "bounds" in annotation_data:
            # Single element annotation
            annotation = self._create_semantic_annotation(annotation_data)
            if annotation:
                annotations.append(annotation)
        else:
            # Try to find any recognized semantic annotation format
            for key, value in annotation_data.items():
                if isinstance(value, dict) and "children" in value:
                    self._process_semantic_node(value, annotations)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "bounds" in item:
                            annotation = self._create_semantic_annotation(item)
                            if annotation:
                                annotations.append(annotation)
        
        return annotations
    
    def _process_semantic_node(self, node: Dict, annotations_list: List):
        """
        Process a semantic annotation node and its children.
        
        Args:
            node (dict): Semantic node
            annotations_list (list): List to append annotations to
        """
        # Process this node
        annotation = self._create_semantic_annotation(node)
        if annotation:
            annotations_list.append(annotation)
        
        # Process children
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                self._process_semantic_node(child, annotations_list)
    
    def _create_semantic_annotation(self, node: Dict) -> Optional[Dict]:
        """
        Create a semantic annotation from a node.
        
        Args:
            node (dict): Semantic node
            
        Returns:
            dict or None: Semantic annotation
        """
        # Skip nodes without bounds
        if "bounds" not in node or not node["bounds"]:
            return None
        
        # Extract component type or class
        component_type = ""
        if "componentLabel" in node:
            component_type = node["componentLabel"]
        elif "class" in node and isinstance(node["class"], str):
            component_type = self._get_component_type_from_class(node["class"])
        
        # Create annotation
        annotation = {
            "type": "component",
            "component_type": component_type,
            "bbox": node["bounds"]
        }
        
        # Add class if available
        if "class" in node and isinstance(node["class"], str):
            annotation["class"] = node["class"]
        
        # Add ancestors if available
        if "ancestors" in node and isinstance(node["ancestors"], list):
            annotation["ancestors"] = node["ancestors"]
        
        # Add clickable if available
        if "clickable" in node:
            annotation["clickable"] = node["clickable"]
        
        return annotation
    
    def _get_component_type_from_class(self, class_name: str) -> str:
        """
        Determine component type from class name.
        
        Args:
            class_name (str): Class name
            
        Returns:
            str: Component type
        """
        class_name = class_name.lower()
        
        # Map common class names to component types
        if "button" in class_name:
            return "Button"
        elif "edittext" in class_name or "input" in class_name:
            return "Text Input"
        elif "imageview" in class_name or "image" in class_name:
            return "Image"
        elif "textview" in class_name or "text" in class_name:
            return "Text"
        elif "checkbox" in class_name:
            return "Checkbox"
        elif "radio" in class_name:
            return "Radio Button"
        elif "spinner" in class_name or "dropdown" in class_name:
            return "Dropdown"
        elif "switch" in class_name:
            return "Switch"
        elif "webview" in class_name:
            return "Web View"
        elif "toolbar" in class_name:
            return "Toolbar"
        elif "menu" in class_name:
            return "Menu"
        elif "list" in class_name:
            return "List"
        elif "grid" in class_name:
            return "Grid"
        elif "card" in class_name:
            return "Card"
        elif "drawer" in class_name:
            return "Navigation Drawer"
        elif "tab" in class_name:
            return "Tab"
        elif "appbar" in class_name or "actionbar" in class_name:
            return "App Bar"
        elif "recycler" in class_name:
            return "Recycler View"
        elif "viewpager" in class_name:
            return "View Pager"
        elif "layout" in class_name:
            return "Layout"
        else:
            return "Unknown"


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = RicoSemanticProcessor(output_dir="./processed_data")
    
    # Process dataset
    processed_data = processor.process(dataset_path="./data/rico_semantics")
    
    print(f"Processed semantic annotations for {len(processed_data)} screens")