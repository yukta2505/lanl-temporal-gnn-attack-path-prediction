#!/usr/bin/env python3

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.universal_log_parser import UniversalLogParser


class TestUniversalLogParser(unittest.TestCase):
    """Test cases for UniversalLogParser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = UniversalLogParser()
        
        # Sample Windows Event Log
        self.windows_event = {
            "EventID": 4624,
            "Computer": "WORKSTATION01",
            "TimeCreated": "2024-01-15T10:30:00.000Z",
            "SubjectUserName": "alice",
            "ProcessName": "winlogon.exe",
            "ProcessId": 1234,
            "CommandLine": "winlogon.exe"
        }
        
        # Sample Sysmon Event
        self.sysmon_event = {
            "EventID": 1,
            "Computer": "WORKSTATION01",
            "UtcTime": "2024-01-15T10:30:00.000Z",
            "User": "DOMAIN\\alice",
            "Image": "C:\\Windows\\System32\\powershell.exe",
            "ProcessId": 5678,
            "CommandLine": "powershell.exe -ExecutionPolicy Bypass",
            "ParentImage": "C:\\Windows\\System32\\cmd.exe",
            "ParentProcessId": 1234
        }
        
        # Sample Linux Audit Event
        self.linux_audit_event = {
            "type": "SYSCALL",
            "@timestamp": "2024-01-15T10:30:00.000Z",
            "hostname": "server01",
            "uid": "1000",
            "exe": "/bin/bash",
            "pid": 9876,
            "proctitle": "bash -c whoami"
        }
        
        # Sample Custom JSON Event
        self.custom_json_event = {
            "timestamp": "2024-01-15T10:30:00.000Z",
            "event_type": "process_create",
            "host": "workstation02",
            "user": "bob",
            "process": "notepad.exe",
            "pid": 4321,
            "command_line": "notepad.exe document.txt"
        }
    
    def test_detect_log_format(self):
        """Test log format detection."""
        # Test Windows Event detection
        self.assertEqual(
            self.parser.detect_log_format(self.windows_event),
            'windows_event'
        )
        
        # Test Sysmon detection
        self.assertEqual(
            self.parser.detect_log_format(self.sysmon_event),
            'sysmon'
        )
        
        # Test Linux Audit detection
        self.assertEqual(
            self.parser.detect_log_format(self.linux_audit_event),
            'linux_audit'
        )
        
        # Test Custom JSON detection
        self.assertEqual(
            self.parser.detect_log_format(self.custom_json_event),
            'custom_json'
        )
    
    def test_parse_windows_event(self):
        """Test Windows Event parsing."""
        normalized = self.parser.parse_windows_event(self.windows_event)
        
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized['event_id'], 4624)
        self.assertEqual(normalized['event_type'], 'user_logon')
        self.assertEqual(normalized['source_host'], 'WORKSTATION01')
        self.assertEqual(normalized['source_user'], 'alice')
        self.assertEqual(normalized['source_process'], 'winlogon.exe')
        self.assertEqual(normalized['source_pid'], 1234)
        self.assertEqual(normalized['timestamp'], "2024-01-15T10:30:00.000Z")
    
    def test_parse_sysmon_event(self):
        """Test Sysmon event parsing."""
        normalized = self.parser.normalize_event(self.sysmon_event)
        
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized['event_id'], 1)
        self.assertEqual(normalized['event_type'], 'process_create')
        self.assertEqual(normalized['source_host'], 'WORKSTATION01')
        self.assertEqual(normalized['source_user'], 'domain\\alice')
        self.assertEqual(normalized['source_process'], 'powershell.exe')
        self.assertEqual(normalized['source_pid'], 5678)
        self.assertEqual(normalized['parent_process'], 'cmd.exe')
        self.assertEqual(normalized['parent_pid'], 1234)
    
    def test_parse_linux_audit_event(self):
        """Test Linux Audit event parsing."""
        normalized = self.parser.normalize_event(self.linux_audit_event)
        
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized['event_type'], 'system_call')
        self.assertEqual(normalized['source_host'], 'SERVER01')
        self.assertEqual(normalized['source_user'], '1000')
        self.assertEqual(normalized['source_process'], 'bash')
        self.assertEqual(normalized['source_pid'], 9876)
    
    def test_parse_custom_json_event(self):
        """Test Custom JSON event parsing."""
        normalized = self.parser.normalize_event(self.custom_json_event)
        
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized['event_type'], 'process_create')
        self.assertEqual(normalized['source_host'], 'WORKSTATION02')
        self.assertEqual(normalized['source_user'], 'bob')
        self.assertEqual(normalized['source_process'], 'notepad.exe')
        self.assertEqual(normalized['source_pid'], 4321)
    
    def test_normalize_event(self):
        """Test event normalization."""
        # Test with different event types
        events = [
            self.windows_event,
            self.sysmon_event,
            self.linux_audit_event,
            self.custom_json_event
        ]
        
        for event in events:
            normalized = self.parser.normalize_event(event)
            self.assertIsNotNone(normalized)
            self.assertIn('timestamp', normalized)
            self.assertIn('event_type', normalized)
            self.assertIn('raw_event', normalized)
    
    def test_normalize_username(self):
        """Test username normalization."""
        # Test domain\user format
        self.assertEqual(
            self.parser.normalize_username('DOMAIN\\Alice'),
            'domain\\alice'
        )
        
        # Test user@domain format
        self.assertEqual(
            self.parser.normalize_username('alice@domain.com'),
            'domain.com\\alice'
        )
        
        # Test plain username
        self.assertEqual(
            self.parser.normalize_username('Alice'),
            'alice'
        )
    
    def test_normalize_hostname(self):
        """Test hostname normalization."""
        # Test with domain
        self.assertEqual(
            self.parser.normalize_hostname('workstation01.domain.com'),
            'WORKSTATION01'
        )
        
        # Test without domain
        self.assertEqual(
            self.parser.normalize_hostname('workstation01'),
            'WORKSTATION01'
        )
    
    def test_normalize_process_name(self):
        """Test process name normalization."""
        # Test full path
        self.assertEqual(
            self.parser.normalize_process_name('C:\\Windows\\System32\\notepad.exe'),
            'notepad.exe'
        )
        
        # Test Linux path
        self.assertEqual(
            self.parser.normalize_process_name('/usr/bin/bash'),
            'bash'
        )
        
        # Test just filename
        self.assertEqual(
            self.parser.normalize_process_name('PowerShell.exe'),
            'powershell.exe'
        )
    
    def test_parse_log_file_json(self):
        """Test parsing JSON log file."""
        # Create temporary JSON file
        events = [self.windows_event, self.sysmon_event]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(events, f)
            temp_file = f.name
        
        try:
            # Parse the file
            normalized_events = self.parser.parse_log_file(temp_file)
            
            self.assertEqual(len(normalized_events), 2)
            self.assertIsNotNone(normalized_events[0]['timestamp'])
            self.assertIsNotNone(normalized_events[1]['timestamp'])
            
        finally:
            os.unlink(temp_file)
    
    def test_parse_log_file_jsonl(self):
        """Test parsing JSONL log file."""
        # Create temporary JSONL file
        events = [self.windows_event, self.sysmon_event]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
            temp_file = f.name
        
        try:
            # Parse the file
            normalized_events = self.parser.parse_log_file(temp_file)
            
            self.assertEqual(len(normalized_events), 2)
            self.assertIsNotNone(normalized_events[0]['timestamp'])
            self.assertIsNotNone(normalized_events[1]['timestamp'])
            
        finally:
            os.unlink(temp_file)
    
    def test_statistics(self):
        """Test parsing statistics."""
        # Reset stats
        self.parser.reset_statistics()
        
        # Parse some events
        self.parser.normalize_event(self.windows_event)
        self.parser.normalize_event(self.sysmon_event)
        self.parser.normalize_event(self.linux_audit_event)
        
        stats = self.parser.get_statistics()
        
        self.assertEqual(stats['total_events_processed'], 3)
        self.assertEqual(stats['windows_events'], 1)
        self.assertEqual(stats['sysmon_events'], 1)
        self.assertEqual(stats['linux_audit_events'], 1)
        self.assertEqual(stats['successful_normalizations'], 3)
    
    def test_performance(self):
        """Test parsing performance with large dataset."""
        import time
        
        # Create 1000 events
        events = [self.windows_event] * 1000
        
        start_time = time.time()
        
        normalized_events = []
        for event in events:
            normalized = self.parser.normalize_event(event)
            if normalized:
                normalized_events.append(normalized)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 events in less than 1 second
        self.assertLess(processing_time, 1.0)
        self.assertEqual(len(normalized_events), 1000)
        
        # Calculate events per second
        events_per_second = len(events) / processing_time
        print(f"\nPerformance: {events_per_second:.0f} events/second")
        
        # Should process at least 1000 events per second
        self.assertGreater(events_per_second, 1000)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty event
        normalized = self.parser.normalize_event({})
        self.assertIsNotNone(normalized)
        
        # Test malformed event
        malformed_event = {"invalid": "data", "no_timestamp": True}
        normalized = self.parser.normalize_event(malformed_event)
        self.assertIsNotNone(normalized)
        
        # Test None values
        event_with_nones = {
            "EventID": None,
            "Computer": None,
            "TimeCreated": None
        }
        normalized = self.parser.normalize_event(event_with_nones)
        self.assertIsNotNone(normalized)
    
    def test_save_normalized_events(self):
        """Test saving normalized events."""
        events = [
            self.parser.normalize_event(self.windows_event),
            self.parser.normalize_event(self.sysmon_event)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save events
            self.parser.save_normalized_events(events, temp_file)
            
            # Verify file exists and contains data
            self.assertTrue(os.path.exists(temp_file))
            
            with open(temp_file, 'r') as f:
                saved_events = json.load(f)
            
            self.assertEqual(len(saved_events), 2)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)