#!/usr/bin/env python3
"""
Entity Extractor for THGNN System

Extracts and normalizes entities (users, hosts, processes, files, registry keys, network connections)
from parsed security events and creates unique ID mappings for graph construction.

Author: THGNN Team
Date: January 2026
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract and normalize entities from security events for graph construction.
    
    Supports extraction of:
    - Users (with domain normalization)
    - Hosts (with FQDN handling)
    - Processes (with path normalization)
    - Files (with path standardization)
    - Registry keys
    - Network connections (IPs, ports)
    """
    
    def __init__(self):
        """Initialize entity extractor with empty collections."""
        # Entity sets
        self.users: Set[str] = set()
        self.hosts: Set[str] = set()
        self.processes: Set[str] = set()
        self.files: Set[str] = set()
        self.registry_keys: Set[str] = set()
        self.network_ips: Set[str] = set()
        self.network_ports: Set[int] = set()
        
        # Entity mappings (bidirectional)
        self.user_to_id: Dict[str, str] = {}
        self.host_to_id: Dict[str, str] = {}
        self.process_to_id: Dict[str, str] = {}
        self.file_to_id: Dict[str, str] = {}
        self.registry_to_id: Dict[str, str] = {}
        
        self.id_to_user: Dict[str, str] = {}
        self.id_to_host: Dict[str, str] = {}
        self.id_to_process: Dict[str, str] = {}
        self.id_to_file: Dict[str, str] = {}
        self.id_to_registry: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'total_events_processed': 0,
            'users_extracted': 0,
            'hosts_extracted': 0,
            'processes_extracted': 0,
            'files_extracted': 0,
            'registry_keys_extracted': 0,
            'network_connections_extracted': 0,
            'extraction_errors': 0
        }
        
        # First/last seen tracking
        self.entity_first_seen: Dict[str, datetime] = {}
        self.entity_last_seen: Dict[str, datetime] = {}
        
    def extract_users(self, events: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique users from events with domain normalization.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Set[str]
            Set of normalized usernames
        """
        try:
            logger.info("Extracting users from events...")
            
            for event in events:
                # Extract source user
                source_user = event.get('source_user')
                if source_user:
                    normalized_user = self.normalize_username(source_user)
                    if normalized_user:
                        self.users.add(normalized_user)
                        self._update_entity_timestamps('user', normalized_user, event.get('timestamp'))
                
                # Extract target user
                target_user = event.get('target_user')
                if target_user:
                    normalized_user = self.normalize_username(target_user)
                    if normalized_user:
                        self.users.add(normalized_user)
                        self._update_entity_timestamps('user', normalized_user, event.get('timestamp'))
            
            self.stats['users_extracted'] = len(self.users)
            logger.info(f"Extracted {len(self.users)} unique users")
            return self.users
            
        except Exception as e:
            logger.error(f"Error extracting users: {e}")
            self.stats['extraction_errors'] += 1
            return set()
    
    def extract_hosts(self, events: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique hosts from events with hostname normalization.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Set[str]
            Set of normalized hostnames
        """
        try:
            logger.info("Extracting hosts from events...")
            
            for event in events:
                # Extract source host
                source_host = event.get('source_host')
                if source_host:
                    normalized_host = self.normalize_hostname(source_host)
                    if normalized_host:
                        self.hosts.add(normalized_host)
                        self._update_entity_timestamps('host', normalized_host, event.get('timestamp'))
                
                # Extract target host
                target_host = event.get('target_host')
                if target_host:
                    normalized_host = self.normalize_hostname(target_host)
                    if normalized_host:
                        self.hosts.add(normalized_host)
                        self._update_entity_timestamps('host', normalized_host, event.get('timestamp'))
                
                # Extract network IPs as potential hosts
                network_src_ip = event.get('network_src_ip')
                if network_src_ip and self._is_valid_ip(network_src_ip):
                    self.hosts.add(network_src_ip.lower())
                    self.network_ips.add(network_src_ip.lower())
                
                network_dst_ip = event.get('network_dst_ip')
                if network_dst_ip and self._is_valid_ip(network_dst_ip):
                    self.hosts.add(network_dst_ip.lower())
                    self.network_ips.add(network_dst_ip.lower())
            
            self.stats['hosts_extracted'] = len(self.hosts)
            logger.info(f"Extracted {len(self.hosts)} unique hosts")
            return self.hosts
            
        except Exception as e:
            logger.error(f"Error extracting hosts: {e}")
            self.stats['extraction_errors'] += 1
            return set()
    
    def extract_processes(self, events: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique processes from events with path normalization.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Set[str]
            Set of normalized process names
        """
        try:
            logger.info("Extracting processes from events...")
            
            for event in events:
                # Extract source process
                source_process = event.get('source_process')
                if source_process:
                    normalized_process = self.normalize_process_name(source_process)
                    if normalized_process:
                        self.processes.add(normalized_process)
                        self._update_entity_timestamps('process', normalized_process, event.get('timestamp'))
                
                # Extract target process
                target_process = event.get('target_process')
                if target_process:
                    normalized_process = self.normalize_process_name(target_process)
                    if normalized_process:
                        self.processes.add(normalized_process)
                        self._update_entity_timestamps('process', normalized_process, event.get('timestamp'))
                
                # Extract parent process
                parent_process = event.get('parent_process')
                if parent_process:
                    normalized_process = self.normalize_process_name(parent_process)
                    if normalized_process:
                        self.processes.add(normalized_process)
                        self._update_entity_timestamps('process', normalized_process, event.get('timestamp'))
            
            self.stats['processes_extracted'] = len(self.processes)
            logger.info(f"Extracted {len(self.processes)} unique processes")
            return self.processes
            
        except Exception as e:
            logger.error(f"Error extracting processes: {e}")
            self.stats['extraction_errors'] += 1
            return set()
    
    def extract_files(self, events: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique file paths from events with path normalization.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Set[str]
            Set of normalized file paths
        """
        try:
            logger.info("Extracting files from events...")
            
            for event in events:
                file_path = event.get('file_path')
                if file_path:
                    normalized_path = self.normalize_file_path(file_path)
                    if normalized_path:
                        self.files.add(normalized_path)
                        self._update_entity_timestamps('file', normalized_path, event.get('timestamp'))
            
            self.stats['files_extracted'] = len(self.files)
            logger.info(f"Extracted {len(self.files)} unique files")
            return self.files
            
        except Exception as e:
            logger.error(f"Error extracting files: {e}")
            self.stats['extraction_errors'] += 1
            return set()
    
    def extract_registry_keys(self, events: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique registry keys from events.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Set[str]
            Set of registry key paths
        """
        try:
            logger.info("Extracting registry keys from events...")
            
            for event in events:
                registry_key = event.get('registry_key')
                if registry_key:
                    normalized_key = self.normalize_registry_key(registry_key)
                    if normalized_key:
                        self.registry_keys.add(normalized_key)
                        self._update_entity_timestamps('registry', normalized_key, event.get('timestamp'))
            
            self.stats['registry_keys_extracted'] = len(self.registry_keys)
            logger.info(f"Extracted {len(self.registry_keys)} unique registry keys")
            return self.registry_keys
            
        except Exception as e:
            logger.error(f"Error extracting registry keys: {e}")
            self.stats['extraction_errors'] += 1
            return set()
    
    def extract_network_connections(self, events: List[Dict[str, Any]]) -> Tuple[Set[str], Set[int]]:
        """
        Extract unique network IPs and ports from events.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Tuple[Set[str], Set[int]]
            Tuple of (IP addresses, ports)
        """
        try:
            logger.info("Extracting network connections from events...")
            
            for event in events:
                # Extract IPs
                src_ip = event.get('network_src_ip')
                if src_ip and self._is_valid_ip(src_ip):
                    self.network_ips.add(src_ip.lower())
                
                dst_ip = event.get('network_dst_ip')
                if dst_ip and self._is_valid_ip(dst_ip):
                    self.network_ips.add(dst_ip.lower())
                
                # Extract ports
                dst_port = event.get('network_dst_port')
                if dst_port and isinstance(dst_port, int) and 1 <= dst_port <= 65535:
                    self.network_ports.add(dst_port)
            
            self.stats['network_connections_extracted'] = len(self.network_ips) + len(self.network_ports)
            logger.info(f"Extracted {len(self.network_ips)} IPs and {len(self.network_ports)} ports")
            return self.network_ips, self.network_ports
            
        except Exception as e:
            logger.error(f"Error extracting network connections: {e}")
            self.stats['extraction_errors'] += 1
            return set(), set()
    
    def build_entity_index(self) -> Dict[str, Dict[str, str]]:
        """
        Create unique ID mappings for all entities.
        
        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary containing all entity mappings
        """
        try:
            logger.info("Building entity index...")
            
            # Create user mappings
            for i, user in enumerate(sorted(self.users)):
                user_id = f"user_{i}"
                self.user_to_id[user] = user_id
                self.id_to_user[user_id] = user
            
            # Create host mappings
            for i, host in enumerate(sorted(self.hosts)):
                host_id = f"host_{i}"
                self.host_to_id[host] = host_id
                self.id_to_host[host_id] = host
            
            # Create process mappings
            for i, process in enumerate(sorted(self.processes)):
                process_id = f"process_{i}"
                self.process_to_id[process] = process_id
                self.id_to_process[process_id] = process
            
            # Create file mappings
            for i, file_path in enumerate(sorted(self.files)):
                file_id = f"file_{i}"
                self.file_to_id[file_path] = file_id
                self.id_to_file[file_id] = file_path
            
            # Create registry mappings
            for i, registry_key in enumerate(sorted(self.registry_keys)):
                registry_id = f"registry_{i}"
                self.registry_to_id[registry_key] = registry_id
                self.id_to_registry[registry_id] = registry_key
            
            logger.info(f"Created mappings for {len(self.users)} users, {len(self.hosts)} hosts, "
                       f"{len(self.processes)} processes, {len(self.files)} files, "
                       f"{len(self.registry_keys)} registry keys")
            
            return {
                'user_to_id': self.user_to_id,
                'host_to_id': self.host_to_id,
                'process_to_id': self.process_to_id,
                'file_to_id': self.file_to_id,
                'registry_to_id': self.registry_to_id,
                'id_to_user': self.id_to_user,
                'id_to_host': self.id_to_host,
                'id_to_process': self.id_to_process,
                'id_to_file': self.id_to_file,
                'id_to_registry': self.id_to_registry
            }
            
        except Exception as e:
            logger.error(f"Error building entity index: {e}")
            self.stats['extraction_errors'] += 1
            return {}
    
    def extract_all_entities(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Extract all entity types from events and build index.
        
        Parameters
        ----------
        events : List[Dict[str, Any]]
            List of normalized events
            
        Returns
        -------
        Dict[str, Dict[str, str]]
            Complete entity mappings
        """
        try:
            logger.info(f"Starting entity extraction from {len(events)} events...")
            self.stats['total_events_processed'] = len(events)
            
            # Extract all entity types
            self.extract_users(events)
            self.extract_hosts(events)
            self.extract_processes(events)
            self.extract_files(events)
            self.extract_registry_keys(events)
            self.extract_network_connections(events)
            
            # Build index
            mappings = self.build_entity_index()
            
            logger.info("Entity extraction completed successfully")
            return mappings
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            self.stats['extraction_errors'] += 1
            return {}
    
    def save_entities(self, output_dir: str) -> bool:
        """
        Save extracted entities and mappings to disk.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save individual entity sets
            entities_dir = output_path / "entities"
            entities_dir.mkdir(exist_ok=True)
            
            with open(entities_dir / "users.json", 'w') as f:
                json.dump(list(self.users), f, indent=2)
            
            with open(entities_dir / "hosts.json", 'w') as f:
                json.dump(list(self.hosts), f, indent=2)
            
            with open(entities_dir / "processes.json", 'w') as f:
                json.dump(list(self.processes), f, indent=2)
            
            with open(entities_dir / "files.json", 'w') as f:
                json.dump(list(self.files), f, indent=2)
            
            with open(entities_dir / "registry_keys.json", 'w') as f:
                json.dump(list(self.registry_keys), f, indent=2)
            
            # Save complete mappings
            mappings = {
                'user_to_id': self.user_to_id,
                'host_to_id': self.host_to_id,
                'process_to_id': self.process_to_id,
                'file_to_id': self.file_to_id,
                'registry_to_id': self.registry_to_id,
                'id_to_user': self.id_to_user,
                'id_to_host': self.id_to_host,
                'id_to_process': self.id_to_process,
                'id_to_file': self.id_to_file,
                'id_to_registry': self.id_to_registry,
                'metadata': {
                    'total_users': len(self.users),
                    'total_hosts': len(self.hosts),
                    'total_processes': len(self.processes),
                    'total_files': len(self.files),
                    'total_registry_keys': len(self.registry_keys),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'stats': self.stats
                }
            }
            
            # Save as pickle for fast loading
            with open(output_path / "entity_mappings.pkl", 'wb') as f:
                pickle.dump(mappings, f)
            
            # Save as JSON for human readability
            with open(output_path / "entity_mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Saved entities to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving entities: {e}")
            return False
    
    def load_entities(self, input_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load entity mappings from disk.
        
        Parameters
        ----------
        input_path : str
            Path to entity_mappings.pkl file
            
        Returns
        -------
        Dict[str, Dict[str, str]]
            Loaded entity mappings
        """
        try:
            with open(input_path, 'rb') as f:
                mappings = pickle.load(f)
            
            # Restore internal state
            self.user_to_id = mappings.get('user_to_id', {})
            self.host_to_id = mappings.get('host_to_id', {})
            self.process_to_id = mappings.get('process_to_id', {})
            self.file_to_id = mappings.get('file_to_id', {})
            self.registry_to_id = mappings.get('registry_to_id', {})
            
            self.id_to_user = mappings.get('id_to_user', {})
            self.id_to_host = mappings.get('id_to_host', {})
            self.id_to_process = mappings.get('id_to_process', {})
            self.id_to_file = mappings.get('id_to_file', {})
            self.id_to_registry = mappings.get('id_to_registry', {})
            
            logger.info(f"Loaded entity mappings from {input_path}")
            return mappings
            
        except Exception as e:
            logger.error(f"Error loading entities: {e}")
            return {}
    
    def get_entity_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics dictionary
        """
        return {
            **self.stats,
            'entity_counts': {
                'users': len(self.users),
                'hosts': len(self.hosts),
                'processes': len(self.processes),
                'files': len(self.files),
                'registry_keys': len(self.registry_keys),
                'network_ips': len(self.network_ips),
                'network_ports': len(self.network_ports)
            },
            'top_entities': {
                'users': list(self.users)[:10],
                'hosts': list(self.hosts)[:10],
                'processes': list(self.processes)[:10]
            }
        }
    
    # Normalization helper methods
    
    def normalize_username(self, username: str) -> Optional[str]:
        """Normalize username format."""
        if not username or not isinstance(username, str):
            return None
        
        username = username.strip()
        if not username:
            return None
        
        # Convert to lowercase for consistency
        username = username.lower()
        
        # Handle different domain formats
        if '\\' in username:
            # DOMAIN\user format - already correct
            return username
        elif '@' in username:
            # user@domain format - convert to DOMAIN\user
            user, domain = username.split('@', 1)
            return f"{domain}\\{user}"
        else:
            # Just username - assume local
            return username
    
    def normalize_hostname(self, hostname: str) -> Optional[str]:
        """Normalize hostname format."""
        if not hostname or not isinstance(hostname, str):
            return None
        
        hostname = hostname.strip()
        if not hostname:
            return None
        
        # Convert to uppercase for consistency
        hostname = hostname.upper()
        
        # Remove domain suffix if present (keep short name)
        if '.' in hostname:
            hostname = hostname.split('.')[0]
        
        return hostname
    
    def normalize_process_name(self, process: str) -> Optional[str]:
        """Normalize process name."""
        if not process or not isinstance(process, str):
            return None
        
        process = process.strip()
        if not process:
            return None
        
        # Extract just the filename from full path
        if '\\' in process:
            process = process.split('\\')[-1]
        elif '/' in process:
            process = process.split('/')[-1]
        
        # Convert to lowercase for consistency
        return process.lower()
    
    def normalize_file_path(self, file_path: str) -> Optional[str]:
        """Normalize file path."""
        if not file_path or not isinstance(file_path, str):
            return None
        
        file_path = file_path.strip()
        if not file_path:
            return None
        
        # Normalize path separators to forward slashes
        file_path = file_path.replace('\\', '/')
        
        # Convert to lowercase
        return file_path.lower()
    
    def normalize_registry_key(self, registry_key: str) -> Optional[str]:
        """Normalize registry key path."""
        if not registry_key or not isinstance(registry_key, str):
            return None
        
        registry_key = registry_key.strip()
        if not registry_key:
            return None
        
        # Convert to uppercase for consistency
        return registry_key.upper()
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Check if string is a valid IP address."""
        if not ip or not isinstance(ip, str):
            return False
        
        # Simple IPv4 validation
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            for part in parts:
                num = int(part)
                if not 0 <= num <= 255:
                    return False
            return True
        except ValueError:
            return False
    
    def _update_entity_timestamps(self, entity_type: str, entity_name: str, timestamp: Optional[str]):
        """Update first/last seen timestamps for entity."""
        if not timestamp:
            return
        
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            entity_key = f"{entity_type}:{entity_name}"
            
            if entity_key not in self.entity_first_seen:
                self.entity_first_seen[entity_key] = dt
            
            self.entity_last_seen[entity_key] = dt
            
        except Exception:
            # Ignore timestamp parsing errors
            pass


def main():
    """Example usage of EntityExtractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entity Extractor for Security Events')
    parser.add_argument('input_file', help='Input file with normalized events (JSON)')
    parser.add_argument('-o', '--output', default='data/processed', help='Output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load events
    try:
        with open(args.input_file, 'r') as f:
            events = json.load(f)
        
        if not isinstance(events, list):
            events = [events]
        
        logger.info(f"Loaded {len(events)} events from {args.input_file}")
        
    except Exception as e:
        logger.error(f"Error loading events: {e}")
        return
    
    # Extract entities
    extractor = EntityExtractor()
    mappings = extractor.extract_all_entities(events)
    
    if mappings:
        # Save results
        success = extractor.save_entities(args.output)
        
        if success:
            # Print statistics
            stats = extractor.get_entity_stats()
            print("\n=== Entity Extraction Statistics ===")
            for key, value in stats['entity_counts'].items():
                print(f"{key}: {value}")
            
            print(f"\n=== Sample Entities ===")
            for entity_type, entities in stats['top_entities'].items():
                print(f"{entity_type}: {entities}")
        
        else:
            logger.error("Failed to save entities")
    else:
        logger.error("Entity extraction failed")


if __name__ == "__main__":
    main()