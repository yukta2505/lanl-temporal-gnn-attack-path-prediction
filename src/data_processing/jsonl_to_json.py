#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartJSONLConverter:
    """Smart converter that checks existing data and processes only if needed."""
    
    def __init__(self, input_dir: str = "data/raw/extracted", 
                 output_dir: str = "data/raw/converted"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converted_files = []
    
    def check_existing_converted_data(self) -> bool:
        """Check if we already have converted data."""
        converted_files = list(self.output_dir.glob("*_converted.json"))
        
        if converted_files:
            logger.info(f"Found {len(converted_files)} already converted files")
            
            # Quick validation of existing files
            valid_files = 0
            total_events = 0
            
            for file_path in converted_files[:5]:  # Check first 5 files
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'events' in data and isinstance(data['events'], list):
                        valid_files += 1
                        total_events += len(data['events'])
                except:
                    continue
            
            if valid_files >= 3 and total_events > 100:
                logger.info(f"Existing converted data looks good: {len(converted_files)} files, {total_events}+ events")
                return True
        
        return False
    
    def find_all_data_files(self) -> List[Path]:
        """Find ALL JSON files recursively."""
        data_files = []
        
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return data_files
        
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.json', '.jsonl')):
                    file_path = Path(root) / file
                    data_files.append(file_path)
        
        logger.info(f"Found {len(data_files)} data files to process")
        return sorted(data_files)
    
    def detect_file_format(self, file_path: Path) -> str:
        """Detect if file is JSON array, JSONL, or single JSON object."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)
                
                if not first_char:
                    return 'empty'
                
                # Check if it starts with [ (JSON array)
                if first_char == '[':
                    return 'json_array'
                
                # Try to parse first line for JSONL or single JSON object
                first_line = f.readline().strip()
                
                if not first_line:
                    return 'empty'
                
                try:
                    parsed = json.loads(first_line)
                    
                    if isinstance(parsed, dict):
                        # Check if there's a second line (JSONL) or not (single JSON object)
                        second_line = f.readline().strip()
                        if second_line:
                            return 'jsonl'
                        else:
                            return 'json_object'
                    else:
                        return 'unknown'
                        
                except json.JSONDecodeError:
                    return 'invalid'
                    
        except Exception as e:
            logger.error(f"Error detecting format for {file_path}: {e}")
            return 'error'
    
    def convert_jsonl_to_array(self, file_path: Path) -> List[Dict]:
        """Convert JSONL file to array of events."""
        events = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict):
                            events.append(event)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num} in {file_path.name}: {e}")
        
        return events
    
    def convert_json_file(self, file_path: Path) -> List[Dict]:
        """Convert JSON file to array of events."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return [event for event in data if isinstance(event, dict)]
            elif isinstance(data, dict):
                if 'events' in data and isinstance(data['events'], list):
                    return data['events']
                else:
                    return [data]
            else:
                logger.warning(f"Unexpected data type in {file_path.name}: {type(data)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path.name}: {e}")
            return []
    
    def extract_metadata(self, events: List[Dict], source_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata from events."""
        if not events:
            return {
                "source_file": str(source_path),
                "total_events": 0,
                "processing_timestamp": datetime.now().isoformat()
            }
        
        timestamps = []
        event_types = set()
        hosts = set()
        users = set()
        processes = set()
        
        for event in events:
            # Extract timestamps
            timestamp_fields = ['@timestamp', 'timestamp', 'TimeCreated', 'UtcTime', 'EventTime']
            for field in timestamp_fields:
                if field in event and event[field]:
                    timestamps.append(str(event[field]))
                    break
            
            # Extract event types
            if 'EventID' in event:
                event_types.add(f"EventID_{event['EventID']}")
            if 'event_type' in event:
                event_types.add(str(event['event_type']))
            if 'Channel' in event:
                event_types.add(f"Channel_{event['Channel']}")
            
            # Extract entities
            host_fields = ['Computer', 'hostname', 'host', 'ComputerName', 'Hostname']
            for field in host_fields:
                if field in event and event[field]:
                    hosts.add(str(event[field]).lower())
                    break
            
            user_fields = ['SubjectUserName', 'TargetUserName', 'User', 'username', 'AccountName']
            for field in user_fields:
                if field in event and event[field]:
                    users.add(str(event[field]).lower())
                    break
            
            process_fields = ['ProcessName', 'NewProcessName', 'Image', 'SourceImage', 'TargetImage']
            for field in process_fields:
                if field in event and event[field]:
                    process_name = Path(str(event[field])).name if '\\' in str(event[field]) else str(event[field])
                    processes.add(process_name.lower())
                    break
        
        metadata = {
            "source_file": str(source_path),
            "total_events": len(events),
            "unique_event_types": len(event_types),
            "unique_hosts": len(hosts),
            "unique_users": len(users),
            "unique_processes": len(processes),
            "event_types": sorted(list(event_types)),
            "hosts": sorted(list(hosts)),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        if timestamps:
            metadata["time_range"] = {
                "start": min(timestamps),
                "end": max(timestamps),
                "span_events": len(timestamps)
            }
        
        return metadata
    
    def convert_file(self, file_path: Path) -> Optional[Path]:
        """Convert single file to standardized JSON format."""
        try:
            file_format = self.detect_file_format(file_path)
            
            if file_format == 'empty':
                logger.warning(f"Skipping empty file: {file_path.name}")
                return None
            elif file_format in ['invalid', 'error']:
                logger.error(f"Skipping invalid file: {file_path.name}")
                return None
            
            # Convert based on format
            if file_format == 'jsonl':
                events = self.convert_jsonl_to_array(file_path)
            elif file_format in ['json_array', 'json_object']:
                events = self.convert_json_file(file_path)
            else:
                logger.warning(f"Unknown format for {file_path.name}: {file_format}")
                return None
            
            if not events:
                logger.warning(f"No valid events found in {file_path.name}")
                return None
            
            # Generate output path
            relative_path = file_path.relative_to(self.input_dir)
            dataset_name = relative_path.parts[0] if len(relative_path.parts) > 1 else file_path.stem
            output_name = f"{dataset_name}_{file_path.stem}_converted.json"
            output_path = self.output_dir / output_name
            
            # Create standardized output
            metadata = self.extract_metadata(events, file_path)
            
            output_data = {
                "metadata": metadata,
                "events": events
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)
            
            self.converted_files.append(str(output_path))
            
            logger.info(f"SUCCESS: {file_path.name} -> {output_path.name} ({len(events)} events)")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {e}")
            return None
    
    def convert_all(self) -> Dict:
        """Convert ALL data files with smart checking."""
        
        # Check if we already have good converted data
        if self.check_existing_converted_data():
            response = input("Converted data already exists and looks good. Skip conversion? (y/n): ").lower()
            if response in ['y', 'yes']:
                logger.info("SKIPPING: Using existing converted data")
                converted_files = list(self.output_dir.glob("*_converted.json"))
                return {
                    'status': 'SKIPPED',
                    'total_files': len(converted_files),
                    'successful_conversions': len(converted_files),
                    'converted_files': [str(f) for f in converted_files]
                }
        
        data_files = self.find_all_data_files()
        
        if not data_files:
            logger.warning("No data files found to convert")
            return {'status': 'NO_FILES', 'results': []}
        
        results = []
        successful = 0
        total_events = 0
        
        for i, data_file in enumerate(data_files, 1):
            logger.info(f"[{i}/{len(data_files)}] Converting {data_file.name}")
            output_path = self.convert_file(data_file)
            
            result = {
                'source_file': str(data_file),
                'output_file': str(output_path) if output_path else None,
                'status': 'SUCCESS' if output_path else 'FAILED'
            }
            
            if output_path:
                successful += 1
                try:
                    with open(output_path, 'r') as f:
                        data = json.load(f)
                        event_count = len(data.get('events', []))
                        result['event_count'] = event_count
                        total_events += event_count
                except:
                    result['event_count'] = 0
            
            results.append(result)
        
        summary = {
            'total_files': len(data_files),
            'successful_conversions': successful,
            'failed_conversions': len(data_files) - successful,
            'success_rate': (successful / len(data_files)) * 100 if data_files else 0,
            'total_events_converted': total_events,
            'converted_files': self.converted_files,
            'results': results
        }
        
        self.save_conversion_manifest(summary)
        
        logger.info(f"CONVERSION COMPLETE: {successful}/{len(data_files)} successful ({summary['success_rate']:.1f}%)")
        logger.info(f"Total events converted: {total_events}")
        
        return summary
    
    def save_conversion_manifest(self, summary: Dict) -> None:
        """Save conversion manifest."""
        manifest_path = Path("data/metadata/conversion_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Conversion manifest saved: {manifest_path}")

def main():
    converter = SmartJSONLConverter()
    summary = converter.convert_all()
    
    if summary.get('successful_conversions', 0) > 0 or summary.get('status') == 'SKIPPED':
        logger.info("Ready for validation step!")
        return True
    else:
        logger.error("No files converted successfully!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
