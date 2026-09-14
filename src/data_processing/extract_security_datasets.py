#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityDatasetsExtractor:
    """Extract JSON files from Security-Datasets repository structure."""
    
    def __init__(self, source_dir: str = "Security-Datasets", 
                 target_dir: str = "data/raw/extracted"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_files = []
        
        # Focus on lateral movement and credential access for THGNN
        self.target_categories = [
            'lateral_movement',
            'credential_access', 
            'execution',
            'defense_evasion',
            'discovery'
        ]
        
    def find_all_json_files(self) -> List[Path]:
        """Find ALL JSON files in Security-Datasets structure."""
        json_files = []
        
        if not self.source_dir.exists():
            logger.error(f"Security-Datasets directory not found: {self.source_dir}")
            return json_files
        
        # Search in datasets/atomic/windows structure
        windows_path = self.source_dir / "datasets" / "atomic" / "windows"
        
        if windows_path.exists():
            for category in self.target_categories:
                category_path = windows_path / category / "host"
                if category_path.exists():
                    for json_file in category_path.glob("*.json"):
                        json_files.append(json_file)
                        
        # Also check for any other JSON files
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith('.json') and 'host' in root:
                    file_path = Path(root) / file
                    if file_path not in json_files:
                        json_files.append(file_path)
        
        logger.info(f"Found {len(json_files)} JSON files in Security-Datasets")
        return sorted(json_files)
    
    def categorize_file(self, file_path: Path) -> str:
        """Categorize file based on path structure."""
        path_parts = file_path.parts
        
        # Extract category from path
        if 'lateral_movement' in path_parts:
            return 'lateral_movement'
        elif 'credential_access' in path_parts:
            return 'credential_access'
        elif 'execution' in path_parts:
            return 'execution'
        elif 'defense_evasion' in path_parts:
            return 'defense_evasion'
        elif 'discovery' in path_parts:
            return 'discovery'
        elif 'persistence' in path_parts:
            return 'persistence'
        elif 'privilege_escalation' in path_parts:
            return 'privilege_escalation'
        else:
            return 'other'
    
    def extract_attack_technique(self, file_path: Path) -> str:
        """Extract attack technique from filename."""
        filename = file_path.stem.lower()
        
        # Common attack techniques
        if 'psexec' in filename:
            return 'psexec'
        elif 'wmi' in filename:
            return 'wmi'
        elif 'dcom' in filename:
            return 'dcom'
        elif 'mimikatz' in filename:
            return 'mimikatz'
        elif 'covenant' in filename:
            return 'covenant'
        elif 'empire' in filename:
            return 'empire'
        elif 'powershell' in filename or 'psh' in filename:
            return 'powershell'
        elif 'cmd' in filename:
            return 'cmd'
        else:
            return 'unknown'
    
    def copy_and_organize_file(self, source_path: Path) -> Optional[Path]:
        """Copy file to organized structure."""
        try:
            category = self.categorize_file(source_path)
            technique = self.extract_attack_technique(source_path)
            
            # Create organized directory structure
            target_category_dir = self.target_dir / category
            target_category_dir.mkdir(exist_ok=True)
            
            # Generate descriptive filename
            original_name = source_path.stem
            new_filename = f"{technique}_{original_name}.json"
            target_path = target_category_dir / new_filename
            
            # Handle filename conflicts
            counter = 1
            while target_path.exists():
                new_filename = f"{technique}_{original_name}_{counter}.json"
                target_path = target_category_dir / new_filename
                counter += 1
            
            # Copy file
            shutil.copy2(source_path, target_path)
            
            logger.info(f"Copied: {source_path.name} -> {category}/{new_filename}")
            
            return target_path
            
        except Exception as e:
            logger.error(f"Failed to copy {source_path}: {e}")
            return None
    
    def validate_json_file(self, file_path: Path) -> Dict:
        """Validate and analyze JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                events = data
            elif isinstance(data, dict) and 'events' in data:
                events = data['events']
            elif isinstance(data, dict):
                events = [data]
            else:
                return {'status': 'invalid', 'error': 'Unknown data format'}
            
            # Analyze events
            event_count = len(events) if isinstance(events, list) else 1
            
            # Sample first event for analysis
            sample_event = events[0] if events else {}
            
            # Check for key fields
            has_timestamp = any(field in sample_event for field in 
                              ['@timestamp', 'timestamp', 'TimeCreated', 'UtcTime'])
            has_host = any(field in sample_event for field in 
                          ['Computer', 'hostname', 'host'])
            has_event_id = 'EventID' in sample_event
            
            return {
                'status': 'valid',
                'event_count': event_count,
                'has_timestamp': has_timestamp,
                'has_host': has_host,
                'has_event_id': has_event_id,
                'sample_fields': list(sample_event.keys())[:10] if sample_event else []
            }
            
        except json.JSONDecodeError as e:
            return {'status': 'invalid', 'error': f'JSON decode error: {e}'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def extract_all(self) -> Dict:
        """Extract and organize all JSON files."""
        json_files = self.find_all_json_files()
        
        if not json_files:
            logger.warning("No JSON files found in Security-Datasets")
            return {'status': 'NO_FILES', 'results': []}
        
        results = []
        successful = 0
        total_events = 0
        categories = {}
        techniques = {}
        
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"[{i}/{len(json_files)}] Processing {json_file.name}")
            
            # Validate file first
            validation = self.validate_json_file(json_file)
            
            if validation['status'] != 'valid':
                logger.warning(f"Skipping invalid file {json_file.name}: {validation.get('error', 'Unknown error')}")
                results.append({
                    'source_file': str(json_file),
                    'status': 'INVALID',
                    'error': validation.get('error', 'Invalid format')
                })
                continue
            
            # Copy and organize file
            target_path = self.copy_and_organize_file(json_file)
            
            if target_path:
                category = self.categorize_file(json_file)
                technique = self.extract_attack_technique(json_file)
                
                # Update counters
                successful += 1
                total_events += validation['event_count']
                categories[category] = categories.get(category, 0) + 1
                techniques[technique] = techniques.get(technique, 0) + 1
                
                result = {
                    'source_file': str(json_file),
                    'target_file': str(target_path),
                    'category': category,
                    'technique': technique,
                    'event_count': validation['event_count'],
                    'status': 'SUCCESS'
                }
                
                self.extracted_files.append(str(target_path))
            else:
                result = {
                    'source_file': str(json_file),
                    'status': 'FAILED',
                    'error': 'Copy failed'
                }
            
            results.append(result)
        
        summary = {
            'total_files': len(json_files),
            'successful_extractions': successful,
            'failed_extractions': len(json_files) - successful,
            'success_rate': (successful / len(json_files)) * 100 if json_files else 0,
            'total_events': total_events,
            'categories': categories,
            'techniques': techniques,
            'extracted_files': self.extracted_files,
            'results': results
        }
        
        self.save_extraction_manifest(summary)
        
        logger.info(f"EXTRACTION COMPLETE: {successful}/{len(json_files)} files ({summary['success_rate']:.1f}%)")
        logger.info(f"Total events: {total_events}")
        logger.info(f"Categories: {dict(categories)}")
        logger.info(f"Techniques: {dict(techniques)}")
        
        return summary
    
    def save_extraction_manifest(self, summary: Dict) -> None:
        """Save extraction manifest."""
        manifest_path = Path("data/metadata/security_datasets_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Manifest saved: {manifest_path}")

def main():
    extractor = SecurityDatasetsExtractor()
    summary = extractor.extract_all()
    
    if summary['successful_extractions'] > 0:
        logger.info("SUCCESS: Security datasets extracted and organized!")
        return True
    else:
        logger.error("FAILED: No files extracted!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
