#!/usr/bin/env python3

import os
import zipfile
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartDatasetExtractor:
    """Smart extractor that checks existing data and processes only if needed."""
    
    def __init__(self, 
                 input_dir: str = "data/raw/mordor_datasets",
                 output_dir: str = "data/raw/extracted"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extraction_log = []
        self.stats = {
            'total_zips_found': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_files_extracted': 0,
            'skipped_existing': 0
        }
    
    def check_existing_extracted_data(self) -> bool:
        """Check if we already have extracted data."""
        extracted_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if len(extracted_dirs) >= 3:  # At least 3 datasets
            total_files = 0
            for dir_path in extracted_dirs:
                json_files = list(dir_path.glob("*.json*"))
                total_files += len(json_files)
            
            if total_files >= 5:  # At least 5 data files
                logger.info(f"Found existing extracted data: {len(extracted_dirs)} datasets, {total_files} files")
                return True
        
        return False
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file for integrity checking."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {filepath}: {e}")
            return ""
    
    def verify_zip_integrity(self, zip_path: Path) -> bool:
        """Verify ZIP file integrity."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test the ZIP file
                bad_file = zip_ref.testzip()
                if bad_file:
                    logger.error(f"Corrupted file in ZIP: {bad_file}")
                    return False
                
                # Check if ZIP has content
                file_list = zip_ref.namelist()
                if not file_list:
                    logger.warning(f"Empty ZIP file: {zip_path.name}")
                    return False
                
                return True
                
        except zipfile.BadZipFile:
            logger.error(f"Bad ZIP file: {zip_path.name}")
            return False
        except Exception as e:
            logger.error(f"Error verifying ZIP {zip_path.name}: {e}")
            return False
    
    def extract_single_zip(self, zip_path: Path) -> Optional[Path]:
        """Extract a single ZIP file with comprehensive error handling."""
        try:
            # Create output directory based on ZIP filename
            dataset_name = zip_path.stem
            extract_dir = self.output_dir / dataset_name
            
            # Check if already extracted
            if extract_dir.exists():
                existing_files = list(extract_dir.glob("*"))
                if len(existing_files) > 0:
                    logger.info(f"SKIPPING: {dataset_name} already extracted ({len(existing_files)} files)")
                    self.stats['skipped_existing'] += 1
                    return extract_dir
            
            # Verify ZIP integrity first
            if not self.verify_zip_integrity(zip_path):
                logger.error(f"FAILED: ZIP integrity check failed for {zip_path.name}")
                return None
            
            # Create extraction directory
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting {zip_path.name} to {extract_dir.name}...")
            
            extracted_files = []
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                for file_info in zip_ref.infolist():
                    # Skip directories
                    if file_info.is_dir():
                        continue
                    
                    # Get clean filename (handle nested paths)
                    filename = Path(file_info.filename).name
                    if not filename:  # Skip if no actual filename
                        continue
                    
                    # Extract file
                    try:
                        # Read file content
                        file_content = zip_ref.read(file_info.filename)
                        
                        # Write to extraction directory
                        output_path = extract_dir / filename
                        with open(output_path, 'wb') as f:
                            f.write(file_content)
                        
                        extracted_files.append(str(output_path))
                        logger.debug(f"  Extracted: {filename} ({len(file_content)} bytes)")
                        
                    except Exception as e:
                        logger.warning(f"  Failed to extract {file_info.filename}: {e}")
                        continue
            
            if extracted_files:
                self.stats['successful_extractions'] += 1
                self.stats['total_files_extracted'] += len(extracted_files)
                
                # Log extraction details
                extraction_record = {
                    'zip_file': str(zip_path),
                    'extract_dir': str(extract_dir),
                    'extracted_files': extracted_files,
                    'file_count': len(extracted_files),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'zip_hash': self.calculate_file_hash(zip_path)
                }
                self.extraction_log.append(extraction_record)
                
                logger.info(f"SUCCESS: Extracted {len(extracted_files)} files from {zip_path.name}")
                return extract_dir
            else:
                logger.error(f"FAILED: No files extracted from {zip_path.name}")
                # Clean up empty directory
                if extract_dir.exists():
                    extract_dir.rmdir()
                return None
                
        except Exception as e:
            logger.error(f"FAILED: Error extracting {zip_path.name}: {e}")
            self.stats['failed_extractions'] += 1
            return None
    
    def handle_nested_zips(self, extract_dir: Path) -> None:
        """Handle nested ZIP files within extracted content."""
        nested_zips = list(extract_dir.glob("*.zip"))
        
        if nested_zips:
            logger.info(f"Found {len(nested_zips)} nested ZIP files in {extract_dir.name}")
            
            for nested_zip in nested_zips:
                logger.info(f"Extracting nested ZIP: {nested_zip.name}")
                
                # Create subdirectory for nested content
                nested_dir = extract_dir / nested_zip.stem
                nested_dir.mkdir(exist_ok=True)
                
                try:
                    with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
                        zip_ref.extractall(nested_dir)
                    
                    # Remove the nested ZIP file after extraction
                    nested_zip.unlink()
                    logger.info(f"  Extracted and removed nested ZIP: {nested_zip.name}")
                    
                except Exception as e:
                    logger.warning(f"  Failed to extract nested ZIP {nested_zip.name}: {e}")
    
    def find_all_zip_files(self) -> List[Path]:
        """Find all ZIP files in the input directory."""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        zip_files = []
        
        # Look for ZIP files recursively
        for zip_path in self.input_dir.rglob("*.zip"):
            if zip_path.is_file():
                zip_files.append(zip_path)
        
        # Also check the Security-Datasets directory if it exists
        security_datasets_dir = Path("Security-Datasets")
        if security_datasets_dir.exists():
            for zip_path in security_datasets_dir.rglob("*.zip"):
                if zip_path.is_file():
                    zip_files.append(zip_path)
        
        self.stats['total_zips_found'] = len(zip_files)
        logger.info(f"Found {len(zip_files)} ZIP files to process")
        
        return sorted(zip_files)
    
    def extract_all(self) -> Dict:
        """Extract all ZIP files with smart checking."""
        
        # Check if we already have extracted data
        if self.check_existing_extracted_data():
            response = input("Extracted data already exists. Skip extraction? (y/n): ").lower()
            if response in ['y', 'yes']:
                logger.info("SKIPPING: Using existing extracted data")
                extracted_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
                return {
                    'status': 'SKIPPED',
                    'total_datasets': len(extracted_dirs),
                    'extraction_log': [{'extract_dir': str(d)} for d in extracted_dirs]
                }
        
        zip_files = self.find_all_zip_files()
        
        if not zip_files:
            logger.warning("No ZIP files found to extract")
            return {'status': 'NO_FILES', 'extraction_log': []}
        
        logger.info(f"Starting extraction of {len(zip_files)} ZIP files...")
        
        successful_extractions = []
        
        for i, zip_path in enumerate(zip_files, 1):
            logger.info(f"[{i}/{len(zip_files)}] Processing {zip_path.name}")
            
            extract_dir = self.extract_single_zip(zip_path)
            
            if extract_dir:
                # Handle nested ZIPs
                self.handle_nested_zips(extract_dir)
                successful_extractions.append(extract_dir)
        
        # Save extraction manifest
        self.save_extraction_manifest()
        
        summary = {
            'total_zips_found': self.stats['total_zips_found'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'skipped_existing': self.stats['skipped_existing'],
            'total_files_extracted': self.stats['total_files_extracted'],
            'success_rate': (self.stats['successful_extractions'] / self.stats['total_zips_found'] * 100) if self.stats['total_zips_found'] > 0 else 0,
            'extraction_log': self.extraction_log
        }
        
        logger.info(f"EXTRACTION COMPLETE:")
        logger.info(f"  ZIP files processed: {self.stats['successful_extractions']}/{self.stats['total_zips_found']} ({summary['success_rate']:.1f}%)")
        logger.info(f"  Total files extracted: {self.stats['total_files_extracted']}")
        logger.info(f"  Datasets ready: {len(successful_extractions)}")
        
        return summary
    
    def save_extraction_manifest(self) -> None:
        """Save extraction manifest for tracking."""
        manifest_path = Path("data/metadata/extraction_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            'extraction_summary': self.stats,
            'extraction_timestamp': datetime.now().isoformat(),
            'extractor_version': '2.0',
            'extraction_log': self.extraction_log
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Extraction manifest saved: {manifest_path}")

def main():
    """Main execution function."""
    extractor = SmartDatasetExtractor()
    
    try:
        summary = extractor.extract_all()
        
        if summary.get('successful_extractions', 0) > 0 or summary.get('status') == 'SKIPPED':
            logger.info("SUCCESS: Data extraction complete! Ready for conversion step.")
            return True
        else:
            logger.error("FAILED: No datasets extracted successfully!")
            return False
            
    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        return False
    except Exception as e:
        logger.error(f"Extraction failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)