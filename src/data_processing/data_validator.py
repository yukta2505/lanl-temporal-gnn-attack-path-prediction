#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartDataValidator:
    """Smart validator that checks existing validation and processes only if needed."""
    
    def __init__(self, data_dir: str = "data/raw/converted"):
        self.data_dir = Path(data_dir)
        self.validation_results = []
    
    def check_existing_validation(self) -> bool:
        """Check if we already have validation results."""
        report_path = Path("data/metadata/validation_report.json")
        
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                summary = report.get('validation_summary', {})
                ready_datasets = summary.get('datasets_ready_for_processing', 0)
                total_events = summary.get('total_valid_events', 0)
                
                if ready_datasets > 0 and total_events > 100:
                    logger.info(f"Found existing validation: {ready_datasets} datasets ready, {total_events} valid events")
                    return True
                    
            except Exception as e:
                logger.warning(f"Could not read existing validation report: {e}")
        
        return False
    
    def validate_timestamp(self, event: Dict) -> bool:
        """Validate timestamp fields."""
        timestamp_fields = ['@timestamp', 'timestamp', 'TimeCreated', 'UtcTime', 'EventTime']
        
        for field in timestamp_fields:
            if field in event and event[field]:
                timestamp = str(event[field])
                if len(timestamp) > 8 and any(char.isdigit() for char in timestamp):
                    return True
        
        return False
    
    def validate_event_structure(self, event: Dict) -> Tuple[bool, List[str]]:
        """Validate event structure and content."""
        issues = []
        
        if not isinstance(event, dict):
            issues.append("Event is not a dictionary")
            return False, issues
        
        if len(event) < 3:
            issues.append("Event has insufficient fields")
        
        if not self.validate_timestamp(event):
            issues.append("Missing or invalid timestamp")
        
        # Check for common required fields
        required_field_groups = [
            ['EventID', 'event_type', 'Channel'],
            ['Computer', 'hostname', 'host', 'ComputerName', 'Hostname'],
        ]
        
        for field_group in required_field_groups:
            if not any(field in event and event[field] for field in field_group):
                issues.append(f"Missing fields from group: {field_group}")
        
        # Check for suspicious content
        event_str = str(event).lower()
        if 'test' in event_str and 'example' in event_str:
            issues.append("Appears to be test/example data")
        
        is_valid = len(issues) <= 1  # Allow 1 minor issue
        return is_valid, issues
    
    def validate_events(self, events: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], Dict]:
        """Validate all events and return valid/invalid splits with stats."""
        valid_events = []
        invalid_events = []
        validation_stats = {
            'total_checked': len(events),
            'timestamp_issues': 0,
            'structure_issues': 0,
            'content_issues': 0,
            'test_data_detected': 0
        }
        
        for i, event in enumerate(events):
            is_valid, issues = self.validate_event_structure(event)
            
            for issue in issues:
                if 'timestamp' in issue.lower():
                    validation_stats['timestamp_issues'] += 1
                elif 'fields' in issue.lower() or 'dictionary' in issue.lower():
                    validation_stats['structure_issues'] += 1
                elif 'test' in issue.lower():
                    validation_stats['test_data_detected'] += 1
                else:
                    validation_stats['content_issues'] += 1
            
            if is_valid:
                valid_events.append(event)
            else:
                invalid_events.append({
                    'event_index': i,
                    'event': event,
                    'issues': issues
                })
        
        return valid_events, invalid_events, validation_stats
    
    def analyze_dataset_quality(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze dataset quality and extract insights."""
        if not events:
            return {'quality_score': 0, 'insights': ['No events to analyze']}
        
        insights = []
        quality_score = 100
        
        # Event type diversity
        event_types = set()
        for event in events:
            if 'EventID' in event:
                event_types.add(f"EventID_{event['EventID']}")
            elif 'event_type' in event:
                event_types.add(str(event['event_type']))
        
        if len(event_types) < 2:
            quality_score -= 15
            insights.append(f"Low event type diversity: {len(event_types)} types")
        else:
            insights.append(f"Good event type diversity: {len(event_types)} types")
        
        # Host diversity
        hosts = set()
        for event in events:
            host_fields = ['Computer', 'hostname', 'host', 'ComputerName', 'Hostname']
            for field in host_fields:
                if field in event and event[field]:
                    hosts.add(str(event[field]).lower())
                    break
        
        if len(hosts) < 1:
            quality_score -= 10
            insights.append("No host information found")
        else:
            insights.append(f"Dataset covers {len(hosts)} hosts")
        
        # Temporal span
        timestamps = []
        for event in events:
            timestamp_fields = ['@timestamp', 'timestamp', 'TimeCreated', 'UtcTime', 'EventTime']
            for field in timestamp_fields:
                if field in event and event[field]:
                    timestamps.append(str(event[field]))
                    break
        
        if len(set(timestamps)) < len(events) * 0.1:
            quality_score -= 10
            insights.append("Low temporal diversity")
        else:
            insights.append("Good temporal spread")
        
        # Data completeness
        complete_events = 0
        for event in events:
            field_count = len([v for v in event.values() if v is not None and str(v).strip()])
            if field_count >= 5:
                complete_events += 1
        
        completeness_rate = complete_events / len(events)
        if completeness_rate < 0.7:
            quality_score -= 20
            insights.append(f"Low data completeness: {completeness_rate:.1%}")
        else:
            insights.append(f"Good data completeness: {completeness_rate:.1%}")
        
        return {
            'quality_score': max(0, quality_score),
            'insights': insights,
            'event_types': sorted(list(event_types)),
            'unique_hosts': len(hosts),
            'temporal_events': len(timestamps),
            'completeness_rate': completeness_rate
        }
    
    def validate_dataset(self, filepath: Path) -> Dict[str, Any]:
        """Validate single dataset with comprehensive analysis."""
        try:
            logger.info(f"Validating {filepath.name}...")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            events = data.get('events', [])
            metadata = data.get('metadata', {})
            
            if not events:
                return {
                    'dataset': filepath.stem,
                    'status': 'EMPTY',
                    'total_events': 0,
                    'validation_timestamp': datetime.now().isoformat()
                }
            
            valid_events, invalid_events, validation_stats = self.validate_events(events)
            quality_analysis = self.analyze_dataset_quality(valid_events)
            
            report = {
                'dataset': filepath.stem,
                'source_file': str(filepath),
                'status': 'SUCCESS',
                'total_events': len(events),
                'valid_events': len(valid_events),
                'invalid_events': len(invalid_events),
                'validity_rate': len(valid_events) / len(events) * 100 if events else 0,
                'validation_stats': validation_stats,
                'quality_analysis': quality_analysis,
                'metadata': metadata,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Save cleaned dataset if needed
            if invalid_events or quality_analysis['quality_score'] < 100:
                cleaned_path = filepath.parent / f"{filepath.stem}_cleaned.json"
                cleaned_data = {
                    "metadata": {
                        **metadata,
                        "cleaned": True,
                        "original_event_count": len(events),
                        "removed_invalid_events": len(invalid_events),
                        "quality_score": quality_analysis['quality_score'],
                        "cleaning_timestamp": datetime.now().isoformat()
                    },
                    "events": valid_events
                }
                
                with open(cleaned_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, indent=2, default=str, ensure_ascii=False)
                
                report['cleaned_file'] = str(cleaned_path)
                logger.info(f"Saved cleaned dataset: {cleaned_path.name}")
            
            logger.info(f"VALIDATED {filepath.name}: {len(valid_events)}/{len(events)} valid ({report['validity_rate']:.1f}%) - Quality: {quality_analysis['quality_score']}/100")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to validate {filepath}: {e}")
            return {
                'dataset': filepath.stem,
                'status': 'ERROR',
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate ALL converted datasets with smart checking."""
        
        # Check if we already have good validation results
        if self.check_existing_validation():
            response = input("Validation already exists and looks good. Skip validation? (y/n): ").lower()
            if response in ['y', 'yes']:
                logger.info("SKIPPING: Using existing validation results")
                with open("data/metadata/validation_report.json", 'r') as f:
                    return json.load(f)
        
        json_files = list(self.data_dir.glob("*_converted.json"))
        
        if not json_files:
            logger.warning(f"No converted JSON files found in {self.data_dir}")
            return {'status': 'NO_FILES'}
        
        logger.info(f"Found {len(json_files)} datasets to validate")
        
        overall_report = {
            "validation_summary": {
                "total_datasets": len(json_files),
                "validation_timestamp": datetime.now().isoformat(),
                "validator_version": "2.0"
            },
            "dataset_reports": {}
        }
        
        total_events = 0
        total_valid = 0
        successful_validations = 0
        quality_scores = []
        
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"[{i}/{len(json_files)}] Validating {json_file.name}")
            report = self.validate_dataset(json_file)
            overall_report["dataset_reports"][json_file.stem] = report
            
            if report['status'] == 'SUCCESS':
                successful_validations += 1
                total_events += report['total_events']
                total_valid += report['valid_events']
                
                if 'quality_analysis' in report:
                    quality_scores.append(report['quality_analysis']['quality_score'])
        
        overall_report["validation_summary"].update({
            "successful_validations": successful_validations,
            "failed_validations": len(json_files) - successful_validations,
            "total_events": total_events,
            "total_valid_events": total_valid,
            "overall_validity_rate": total_valid / total_events * 100 if total_events > 0 else 0,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "datasets_ready_for_processing": len([r for r in overall_report["dataset_reports"].values() 
                                                if r.get('status') == 'SUCCESS' and r.get('valid_events', 0) > 0])
        })
        
        report_path = Path("data/metadata/validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(overall_report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved: {report_path}")
        
        summary = overall_report["validation_summary"]
        logger.info(f"VALIDATION COMPLETE:")
        logger.info(f"  Datasets processed: {summary['successful_validations']}/{summary['total_datasets']}")
        logger.info(f"  Events validated: {summary['total_valid_events']}/{summary['total_events']} ({summary['overall_validity_rate']:.1f}%)")
        logger.info(f"  Average quality: {summary['average_quality_score']:.1f}/100")
        logger.info(f"  Ready for processing: {summary['datasets_ready_for_processing']} datasets")
        
        return overall_report

def main():
    validator = SmartDataValidator()
    report = validator.validate_all()
    
    if report.get('validation_summary', {}).get('datasets_ready_for_processing', 0) > 0:
        logger.info("SUCCESS: Data ready for Day 2 processing!")
        return True
    else:
        logger.error("FAILED: No datasets ready for processing!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
