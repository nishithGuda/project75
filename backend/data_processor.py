#!/usr/bin/env python3
"""
Main script for processing and integrating Rico datasets for the LLM-based UI Navigation Assistant.
This script coordinates the processing of the Rico datasets:
1. Rico Base (Screenshots and View Hierarchies)
2. Rico Semantic Annotations
"""

import os
import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to the path so modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rico_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process and integrate Rico datasets')
    
    parser.add_argument('--rico-base', type=str, help='Path to Rico base dataset (screenshots and view hierarchies)')
    parser.add_argument('--rico-semantics', type=str, help='Path to Rico semantic annotations')
    parser.add_argument('--use-refexp', action='store_true', help='Use RefExp dataset from Hugging Face')
    parser.add_argument('--output-dir', type=str, default='./processed_data', help='Output directory for processed data')
    parser.add_argument('--copy-to-project', action='store_true', help='Copy processed data to project structure')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of screens to process')
    
    return parser.parse_args()

def main():
    """Main function to process and integrate datasets."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Try to import processors
        try:
            from utils.metadata_parser import RicoMetadataProcessor
            from utils.semantic_parser import RicoSemanticProcessor
            from utils.refexp_parser import RicoRefExpProcessor
            from utils.dataset_integrator import DatasetIntegrator
        except ImportError:
            logger.warning("Could not import from utils package, trying direct imports")
            try:
                # Try importing from current directory
                from metadata_parser import RicoMetadataProcessor
                from semantic_parser import RicoSemanticProcessor
                from refexp_parser import RicoRefExpProcessor
                from dataset_integrator import DatasetIntegrator
            except ImportError:
                logger.error("Failed to import processor modules. Make sure they are in the correct path.")
                return
        
        # Initialize processors
        metadata_processor = RicoMetadataProcessor(output_dir=output_dir)
        semantic_processor = RicoSemanticProcessor(output_dir=output_dir)
        refexp_processor = RicoRefExpProcessor(output_dir=output_dir)
        
        # Process datasets
        processed_metadata = None
        processed_semantics = None
        processed_refexp = None
        
        if args.rico_base:
            logger.info(f"Processing Rico base dataset from: {args.rico_base}")
            processed_metadata = metadata_processor.process(args.rico_base, sample_size=args.sample_size)
        
        if args.rico_semantics:
            logger.info(f"Processing Rico semantic annotations from: {args.rico_semantics}")
            processed_semantics = semantic_processor.process(args.rico_semantics)
        
        if args.use_refexp:
            logger.info("Processing Rico RefExp dataset from Hugging Face")
            processed_refexp = refexp_processor.process_from_huggingface()
        
        # Integrate datasets
        integrator = DatasetIntegrator(output_dir=output_dir)
        
        metadata_path = output_dir / "processed" / "base" / "processed_metadata.json"
        semantics_path = output_dir / "processed" / "semantics" / "processed_semantics.json"
        refexp_path = output_dir / "processed" / "refexp" / "processed_refexp.json"
        
        # Check if files exist
        if processed_metadata is None and metadata_path.exists():
            logger.info(f"Using previously processed metadata from {metadata_path}")
            metadata_path = str(metadata_path)
        else:
            metadata_path = processed_metadata
        
        if processed_semantics is None and semantics_path.exists():
            logger.info(f"Using previously processed semantics from {semantics_path}")
            semantics_path = str(semantics_path)
        else:
            semantics_path = processed_semantics
        
        if processed_refexp is None and refexp_path.exists():
            logger.info(f"Using previously processed RefExp data from {refexp_path}")
            refexp_path = str(refexp_path)
        else:
            refexp_path = processed_refexp
        
        # Integrate
        integrated_data = integrator.integrate(
            metadata=metadata_path,
            semantics=semantics_path,
            refexp=refexp_path
        )
        
        # Generate data for project
        integrator.generate_project_data(integrated_data)
        
        # Copy to project structure if requested
        if args.copy_to_project:
            logger.info("Copying data to project structure")
            integrator.copy_to_project()
        
        logger.info(f"Processing complete. Output saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing datasets: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()