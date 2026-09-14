#!/usr/bin/env python3

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.entity_extractor import EntityExtractor


class TestEntityExtractor(unittest.TestCase):
    """Test cases for EntityExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EntityExtractor()
        
        # Sample normalized events
        self.sample_events = [
            {
                "timestamp": "2024-01-15T10:30:00.000Z",
                "event_type": "user_logon",
                "source_host": "WORKSTATION01",
                "source_user": "DOMAIN\\alice",
                "source_process": "winlogon.exe",
                "source_pid": 1234,
                "target_host": "DC01",
                "command_line": "winlogon.exe"
            },
            {
                "timestamp": "2024-01-15T10:31:00.000Z",
                "event_type": "process_create",
                "source_host": "WORKSTATION01",
                "source_user": "DOMAIN\\alice",
                "source_process": "C:\\Windows\\System32\\powershell.exe",
                "source_pid": 5678,
                "target_process": "C:\\Windows\\System32\\cmd.exe",
                "target_pid": 9012,
                "command_line": "powershell.exe -ExecutionPolicy Bypass",
                "parent_process": "C:\\Windows\\explorer.exe",
                "parent_pid": 1234
            },
            {
                "timestamp": "2024-01-15T10:32:00.000Z",
                "event_type": "network_connect",
                "source_host": "WORKSTATION01",
                "source_process": "powershell.exe",
                "source_pid": 5678,
                "network_src_ip": "192.168.1.100",
                "network_dst_ip": "192.168.1.10",
                "network_dst_port": 445
            },
            {
                "timestamp": "2024-01-15T10:33:00.000Z",
                "event_type": "file_access",
                "source_host": "WORKSTATION01",
                "source_process": "powershell.exe",
                "file_path": "C:\\Windows\\System32\\config\\SAM"
            },
            {
                "timestamp": "2024-01-15T10:34:00.000Z",
                "event_type": "registry_modify",
                "source_host": "WORKSTATION01",
                "source_process": "powershell.exe",
                "registry_key": "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
            }
        ]
    
    def test_extract_users(self):
        """Test user extraction."""
        users = self.extractor.extract_users(self.sample_events)
        
        self.assertIn("domain\\alice", users)
        self.assertEqual(len(users), 1)
        self.assertEqual(self.extractor.stats['users_extracted'], 1)
    
    def test_extract_hosts(self):
        """Test host extraction."""
        hosts = self.extractor.extract_hosts(self.sample_events)
        
        expected_hosts = {"WORKSTATION01", "DC01", "192.168.1.100", "192.168.1.10"}
        self.assertTrue(expected_hosts.issubset(hosts))
        self.assertGreaterEqual(len(hosts), 4)
    
    def test_extract_processes(self):
        """Test process extraction."""
        processes = self.extractor.extract_processes(self.sample_events)
        
        expected_processes = {"winlogon.exe", "powershell.exe", "cmd.exe", "explorer.exe"}
        self.assertTrue(expected_processes.issubset(processes))
        self.assertGreaterEqual(len(processes), 4)
    
    def test_extract_files(self):
        """Test file extraction."""
        files = self.extractor.extract_files(self.sample_events)
        
        self.assertIn("c:/windows/system32/config/sam", files)
        self.assertEqual(len(files), 1)
    
    def test_extract_registry_keys(self):
        """Test registry key extraction."""
        registry_keys = self.extractor.extract_registry_keys(self.sample_events)
        
        self.assertIn("HKLM\\SOFTWARE\\MICROSOFT\\WINDOWS\\CURRENTVERSION\\RUN", registry_keys)
        self.assertEqual(len(registry_keys), 1)
    
    def test_extract_network_connections(self):
        """Test network connection extraction."""
        ips, ports = self.extractor.extract_network_connections(self.sample_events)
        
        self.assertIn("192.168.1.100", ips)
        self.assertIn("192.168.1.10", ips)
        self.assertIn(445, ports)
    
    def test_normalize_username(self):
        """Test username normalization."""
        # Test domain\user format
        self.assertEqual(
            self.extractor.normalize_username('DOMAIN\\Alice'),
            'domain\\alice'
        )
        
        # Test user@domain format
        self.assertEqual(
            self.extractor.normalize_username('alice@domain.com'),
            'domain.com\\alice'
        )
        
        # Test plain username
        self.assertEqual(
            self.extractor.normalize_username('Alice'),
            'alice'
        )
        
        # Test invalid input
        self.assertIsNone(self.extractor.normalize_username(''))
        self.assertIsNone(self.extractor.normalize_username(None))
    
    def test_normalize_hostname(self):
        """Test hostname normalization."""
        # Test with domain
        self.assertEqual(
            self.extractor.normalize_hostname('workstation01.domain.com'),
            'WORKSTATION01'
        )
        
        # Test without domain
        self.assertEqual(
            self.extractor.normalize_hostname('workstation01'),
            'WORKSTATION01'
        )
        
        # Test invalid input
        self.assertIsNone(self.extractor.normalize_hostname(''))
        self.assertIsNone(self.extractor.normalize_hostname(None))
    
    def test_normalize_process_name(self):
        """Test process name normalization."""
        # Test full Windows path
        self.assertEqual(
            self.extractor.normalize_process_name('C:\\Windows\\System32\\notepad.exe'),
            'notepad.exe'
        )
        
        # Test Linux path
        self.assertEqual(
            self.extractor.normalize_process_name('/usr/bin/bash'),
            'bash'
        )
        
        # Test just filename
        self.assertEqual(
            self.extractor.normalize_process_name('PowerShell.exe'),
            'powershell.exe'
        )
        
        # Test invalid input
        self.assertIsNone(self.extractor.normalize_process_name(''))
        self.assertIsNone(self.extractor.normalize_process_name(None))
    
    def test_normalize_file_path(self):
        """Test file path normalization."""
        # Test Windows path
        self.assertEqual(
            self.extractor.normalize_file_path('C:\\Windows\\System32\\config\\SAM'),
            'c:/windows/system32/config/sam'
        )
        
        # Test Linux path
        self.assertEqual(
            self.extractor.normalize_file_path('/etc/passwd'),
            '/etc/passwd'
        )
    
    def test_normalize_registry_key(self):
        """Test registry key normalization."""
        self.assertEqual(
            self.extractor.normalize_registry_key('hklm\\software\\microsoft'),
            'HKLM\\SOFTWARE\\MICROSOFT'
        )
    
    def test_is_valid_ip(self):
        """Test IP address validation."""
        # Valid IPs
        self.assertTrue(self.extractor._is_valid_ip('192.168.1.1'))
        self.assertTrue(self.extractor._is_valid_ip('10.0.0.1'))
        self.assertTrue(self.extractor._is_valid_ip('172.16.0.1'))
        
        # Invalid IPs
        self.assertFalse(self.extractor._is_valid_ip('256.1.1.1'))
        self.assertFalse(self.extractor._is_valid_ip('192.168.1'))
        self.assertFalse(self.extractor._is_valid_ip('not.an.ip.address'))
        self.assertFalse(self.extractor._is_valid_ip(''))
        self.assertFalse(self.extractor._is_valid_ip(None))
    
    def test_build_entity_index(self):
        """Test entity index building."""
        # First extract entities
        self.extractor.extract_all_entities(self.sample_events)
        
        # Build index
        mappings = self.extractor.build_entity_index()
        
        # Check mappings exist
        self.assertIn('user_to_id', mappings)
        self.assertIn('host_to_id', mappings)
        self.assertIn('process_to_id', mappings)
        
        # Check bidirectional mappings
        for user, user_id in self.extractor.user_to_id.items():
            self.assertEqual(self.extractor.id_to_user[user_id], user)
        
        for host, host_id in self.extractor.host_to_id.items():
            self.assertEqual(self.extractor.id_to_host[host_id], host)
        
        for process, process_id in self.extractor.process_to_id.items():
            self.assertEqual(self.extractor.id_to_process[process_id], process)
    
    def test_extract_all_entities(self):
        """Test complete entity extraction."""
        mappings = self.extractor.extract_all_entities(self.sample_events)
        
        # Check all mapping types exist
        self.assertIn('user_to_id', mappings)
        self.assertIn('host_to_id', mappings)
        self.assertIn('process_to_id', mappings)
        self.assertIn('file_to_id', mappings)
        self.assertIn('registry_to_id', mappings)
        
        # Check statistics
        stats = self.extractor.get_entity_stats()
        self.assertEqual(stats['total_events_processed'], len(self.sample_events))
        self.assertGreater(stats['entity_counts']['users'], 0)
        self.assertGreater(stats['entity_counts']['hosts'], 0)
        self.assertGreater(stats['entity_counts']['processes'], 0)
    
    def test_save_and_load_entities(self):
        """Test saving and loading entities."""
        # Extract entities
        self.extractor.extract_all_entities(self.sample_events)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            success = self.extractor.save_entities(temp_dir)
            self.assertTrue(success)
            
            # Check files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'entity_mappings.pkl')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'entity_mappings.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'entities', 'users.json')))
            
            # Load entities
            new_extractor = EntityExtractor()
            loaded_mappings = new_extractor.load_entities(
                os.path.join(temp_dir, 'entity_mappings.pkl')
            )
            
            # Check loaded mappings match original
            self.assertEqual(loaded_mappings['user_to_id'], self.extractor.user_to_id)
            self.assertEqual(loaded_mappings['host_to_id'], self.extractor.host_to_id)
            self.assertEqual(loaded_mappings['process_to_id'], self.extractor.process_to_id)
    
    def test_get_entity_stats(self):
        """Test statistics generation."""
        self.extractor.extract_all_entities(self.sample_events)
        stats = self.extractor.get_entity_stats()
        
        # Check required fields
        self.assertIn('entity_counts', stats)
        self.assertIn('top_entities', stats)
        self.assertIn('total_events_processed', stats)
        
        # Check counts are reasonable
        self.assertGreater(stats['entity_counts']['users'], 0)
        self.assertGreater(stats['entity_counts']['hosts'], 0)
        self.assertGreater(stats['entity_counts']['processes'], 0)
    
    def test_performance(self):
        """Test extraction performance with larger dataset."""
        import time
        
        # Create 1000 events
        large_events = self.sample_events * 200  # 1000 events
        
        start_time = time.time()
        mappings = self.extractor.extract_all_entities(large_events)
        end_time = time.time()
        
        processing_time = end_time - start_time
        events_per_second = len(large_events) / processing_time
        
        # Should process at least 1000 events per second
        self.assertGreater(events_per_second, 1000)
        self.assertIsNotNone(mappings)
        
        print(f"\nPerformance: {events_per_second:.0f} events/second")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty events
        mappings = self.extractor.extract_all_entities([])
        self.assertIsNotNone(mappings)
        
        # Test malformed events
        malformed_events = [
            {"invalid": "event"},
            {"timestamp": None, "source_user": ""},
            {}
        ]
        mappings = self.extractor.extract_all_entities(malformed_events)
        self.assertIsNotNone(mappings)
        
        # Test events with None values
        none_events = [
            {
                "timestamp": "2024-01-15T10:30:00.000Z",
                "source_user": None,
                "source_host": None,
                "source_process": None
            }
        ]
        mappings = self.extractor.extract_all_entities(none_events)
        self.assertIsNotNone(mappings)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)