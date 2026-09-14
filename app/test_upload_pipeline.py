#!/usr/bin/env python3
"""
Professional Upload Pipeline Test
Test all file formats and error handling scenarios.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - PROFESSIONAL TEST
"""

import json
import csv
import tempfile
import os
from pathlib import Path
from streamlit_app_perfect import ProfessionalDataProcessor

def create_test_files():
    """Create test files for all supported formats"""
    
    # Sample security event data
    sample_events = [
        {
            "timestamp": "2026-01-10T10:30:00Z",
            "event_id": 4624,
            "user": "DOMAIN\\alice",
            "host": "WORKSTATION01",
            "process": "winlogon.exe",
            "action": "logon",
            "source_ip": "192.168.1.100"
        },
        {
            "timestamp": "2026-01-10T10:31:00Z",
            "event_id": 4688,
            "user": "DOMAIN\\alice",
            "host": "WORKSTATION01",
            "process": "powershell.exe",
            "command": "Get-Process",
            "parent_process": "explorer.exe"
        },
        {
            "timestamp": "2026-01-10T10:32:00Z",
            "event_id": 5156,
            "user": "DOMAIN\\alice",
            "host": "WORKSTATION01",
            "process": "powershell.exe",
            "target_host": "DC01",
            "target_port": 445,
            "protocol": "TCP"
        }
    ]
    
    test_files = {}
    
    # Create JSON test file
    json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(sample_events, json_file, indent=2)
    json_file.close()
    test_files['json'] = json_file.name
    
    # Create JSONL test file
    jsonl_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for event in sample_events:
        jsonl_file.write(json.dumps(event) + '\n')
    jsonl_file.close()
    test_files['jsonl'] = jsonl_file.name
    
    # Create CSV test file
    csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    fieldnames = ['timestamp', 'event_id', 'user', 'host', 'process', 'action', 'source_ip', 'command', 'parent_process', 'target_host', 'target_port', 'protocol']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for event in sample_events:
        # Fill missing fields with empty strings
        row = {field: event.get(field, '') for field in fieldnames}
        writer.writerow(row)
    csv_file.close()
    test_files['csv'] = csv_file.name
    
    # Create invalid JSON file for error testing
    invalid_json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    invalid_json_file.write('{"invalid": json syntax}')
    invalid_json_file.close()
    test_files['invalid_json'] = invalid_json_file.name
    
    return test_files

def test_file_validation():
    """Test file validation functionality"""
    print("Testing file validation...")
    
    processor = ProfessionalDataProcessor()
    
    # Test with None file
    is_valid, message = processor.validate_file(None)
    assert not is_valid
    assert "No file provided" in message
    print("✓ None file validation passed")
    
    # Test with mock file object
    class MockFile:
        def __init__(self, name, size):
            self.name = name
            self.size = size
    
    # Test unsupported format
    mock_file = MockFile("test.txt", 1000)
    is_valid, message = processor.validate_file(mock_file)
    assert not is_valid
    assert "Unsupported format" in message
    print("✓ Unsupported format validation passed")
    
    # Test file too large
    mock_file = MockFile("test.json", 200 * 1024 * 1024)  # 200MB
    is_valid, message = processor.validate_file(mock_file)
    assert not is_valid
    assert "File too large" in message
    print("✓ File size validation passed")
    
    # Test valid file
    mock_file = MockFile("test.json", 1000)
    is_valid, message = processor.validate_file(mock_file)
    assert is_valid
    assert "validation passed" in message
    print("✓ Valid file validation passed")

def test_json_processing():
    """Test JSON file processing"""
    print("\nTesting JSON processing...")
    
    processor = ProfessionalDataProcessor()
    test_files = create_test_files()
    
    # Test valid JSON
    class MockFile:
        def __init__(self, filepath):
            self.filepath = filepath
            self.name = os.path.basename(filepath)
            self.size = os.path.getsize(filepath)
        
        def read(self):
            with open(self.filepath, 'r') as f:
                return f.read()
    
    mock_file = MockFile(test_files['json'])
    success, data, message = processor.process_json_file(mock_file)
    
    assert success
    assert 'events' in data
    assert len(data['events']) == 3
    assert 'stats' in data
    print("✓ Valid JSON processing passed")
    
    # Test invalid JSON
    mock_file = MockFile(test_files['invalid_json'])
    success, data, message = processor.process_json_file(mock_file)
    
    assert not success
    assert "JSON parsing error" in message
    print("✓ Invalid JSON handling passed")
    
    # Cleanup
    for filepath in test_files.values():
        os.unlink(filepath)

def test_jsonl_processing():
    """Test JSONL file processing"""
    print("\nTesting JSONL processing...")
    
    processor = ProfessionalDataProcessor()
    test_files = create_test_files()
    
    class MockFile:
        def __init__(self, filepath):
            self.filepath = filepath
            self.name = os.path.basename(filepath)
            self.size = os.path.getsize(filepath)
        
        def read(self):
            with open(self.filepath, 'r') as f:
                return f.read()
    
    mock_file = MockFile(test_files['jsonl'])
    success, data, message = processor.process_jsonl_file(mock_file)
    
    assert success
    assert 'events' in data
    assert len(data['events']) == 3
    assert 'stats' in data
    print("✓ Valid JSONL processing passed")
    
    # Cleanup
    for filepath in test_files.values():
        os.unlink(filepath)

def test_csv_processing():
    """Test CSV file processing"""
    print("\nTesting CSV processing...")
    
    processor = ProfessionalDataProcessor()
    test_files = create_test_files()
    
    class MockFile:
        def __init__(self, filepath):
            self.filepath = filepath
            self.name = os.path.basename(filepath)
            self.size = os.path.getsize(filepath)
        
        def read(self):
            with open(self.filepath, 'rb') as f:
                return f.read()
    
    # For CSV, we need to simulate the file object properly
    import io
    
    with open(test_files['csv'], 'r') as f:
        csv_content = f.read()
    
    mock_file = io.StringIO(csv_content)
    mock_file.name = "test.csv"
    mock_file.size = len(csv_content)
    
    success, data, message = processor.process_csv_file(mock_file)
    
    assert success
    assert 'events' in data
    assert len(data['events']) == 3
    assert 'stats' in data
    print("✓ Valid CSV processing passed")
    
    # Cleanup
    for filepath in test_files.values():
        os.unlink(filepath)

def test_entity_extraction():
    """Test entity extraction functionality"""
    print("\nTesting entity extraction...")
    
    processor = ProfessionalDataProcessor()
    
    sample_events = [
        {
            "user": "DOMAIN\\alice",
            "host": "WORKSTATION01",
            "process": "powershell.exe"
        },
        {
            "username": "bob",
            "computer": "SERVER01",
            "image": "cmd.exe"
        },
        {
            "account": "charlie",
            "hostname": "DC01",
            "command": "whoami.exe"
        }
    ]
    
    entities = processor.extract_entities(sample_events)
    
    assert 'users' in entities
    assert 'hosts' in entities
    assert 'processes' in entities
    assert len(entities['users']) >= 3
    assert len(entities['hosts']) >= 3
    assert len(entities['processes']) >= 3
    print("✓ Entity extraction passed")

def test_full_pipeline():
    """Test complete processing pipeline"""
    print("\nTesting full processing pipeline...")
    
    processor = ProfessionalDataProcessor()
    test_files = create_test_files()
    
    class MockFile:
        def __init__(self, filepath):
            self.filepath = filepath
            self.name = os.path.basename(filepath)
            self.size = os.path.getsize(filepath)
        
        def read(self):
            with open(self.filepath, 'r') as f:
                return f.read()
    
    # Test JSON pipeline
    mock_file = MockFile(test_files['json'])
    success, data, message = processor.process_file(mock_file)
    
    assert success
    assert 'events' in data
    assert 'entities' in data
    assert 'stats' in data
    assert data['entities']['total_entities'] > 0
    print("✓ Full JSON pipeline passed")
    
    # Test JSONL pipeline
    mock_file = MockFile(test_files['jsonl'])
    success, data, message = processor.process_file(mock_file)
    
    assert success
    assert 'events' in data
    assert 'entities' in data
    print("✓ Full JSONL pipeline passed")
    
    # Cleanup
    for filepath in test_files.values():
        os.unlink(filepath)

def run_all_tests():
    """Run all upload pipeline tests"""
    print("🚀 Starting Professional Upload Pipeline Tests")
    print("=" * 60)
    
    try:
        test_file_validation()
        test_json_processing()
        test_jsonl_processing()
        test_csv_processing()
        test_entity_extraction()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - Upload pipeline is perfect!")
        print("✓ File validation working")
        print("✓ JSON processing working")
        print("✓ JSONL processing working")
        print("✓ CSV processing working")
        print("✓ Entity extraction working")
        print("✓ Full pipeline working")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)