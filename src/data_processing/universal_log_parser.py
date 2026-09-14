#!/usr/bin/env python3

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalLogParser:
    """
    Universal parser for security logs supporting multiple formats:
    - Windows Event Logs
    - Sysmon Events  
    - Linux Audit Logs
    - Custom JSON formats
    
    Normalizes all events to a standard schema for THGNN processing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize universal log parser.
        
        Parameters
        ----------
        config_path : str, optional
            Path to field mappings configuration file
        """
        self.config_path = config_path or "config/field_mappings.yaml"
        self.field_mappings = self.load_field_mappings()
        
        # Statistics
        self.stats = {
            'total_events_processed': 0,
            'windows_events': 0,
            'sysmon_events': 0,
            'linux_audit_events': 0,
            'custom_json_events': 0,
            'parsing_errors': 0,
            'normalization_errors': 0,
            'successful_normalizations': 0
        }
        
        # Normalized schema template
        self.normalized_schema = {
            "timestamp": None,
            "event_type": None,
            "source_host": None,
            "source_user": None,
            "source_process": None,
            "source_pid": None,
            "target_host": None,
            "target_user": None,
            "target_process": None,
            "target_pid": None,
            "network_src_ip": None,
            "network_dst_ip": None,
            "network_dst_port": None,
            "file_path": None,
            "registry_key": None,
            "command_line": None,
            "parent_process": None,
            "parent_pid": None,
            "event_id": None,
            "raw_event": {}
        }
    
    def load_field_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load field mappings from configuration file."""
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {config_path}. Using default mappings.")
                return self.get_default_mappings()
        except Exception as e:
            logger.error(f"Error loading field mappings: {e}. Using defaults.")
            return self.get_default_mappings()
    
    def get_default_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get default field mappings for common log formats."""
        return {
            "windows_event": {
                "timestamp": ["TimeCreated", "@timestamp", "timestamp"],
                "event_id": ["EventID"],
                "source_host": ["Computer", "ComputerName"],
                "source_user": ["SubjectUserName", "TargetUserName", "User"],
                "source_process": ["ProcessName", "NewProcessName"],
                "source_pid": ["ProcessId", "NewProcessId"],
                "target_user": ["TargetUserName", "AccountName"],
                "command_line": ["CommandLine", "ProcessCommandLine"],
                "parent_process": ["ParentProcessName"],
                "parent_pid": ["ParentProcessId"]
            },
            "sysmon": {
                "timestamp": ["UtcTime", "@timestamp", "timestamp"],
                "event_id": ["EventID"],
                "source_host": ["Computer", "ComputerName"],
                "source_user": ["User"],
                "source_process": ["Image", "ProcessName"],
                "source_pid": ["ProcessId"],
                "target_process": ["TargetImage"],
                "target_pid": ["TargetProcessId"],
                "network_src_ip": ["SourceIp"],
                "network_dst_ip": ["DestinationIp"],
                "network_dst_port": ["DestinationPort"],
                "file_path": ["TargetFilename", "ImageLoaded"],
                "registry_key": ["TargetObject"],
                "command_line": ["CommandLine"],
                "parent_process": ["ParentImage"],
                "parent_pid": ["ParentProcessId"]
            },
            "linux_audit": {
                "timestamp": ["@timestamp", "timestamp"],
                "event_type": ["type"],
                "source_host": ["hostname", "host"],
                "source_user": ["uid", "auid"],
                "source_process": ["exe", "comm"],
                "source_pid": ["pid"],
                "file_path": ["name", "path"],
                "command_line": ["proctitle"]
            },
            "custom_json": {
                "timestamp": ["timestamp", "@timestamp", "time", "event_time"],
                "event_type": ["event_type", "type", "action"],
                "source_host": ["host", "hostname", "computer", "source_host"],
                "source_user": ["user", "username", "account", "source_user"],
                "source_process": ["process", "process_name", "image", "source_process"],
                "source_pid": ["pid", "process_id", "source_pid"],
                "target_host": ["target_host", "destination_host", "dest_host"],
                "target_user": ["target_user", "destination_user", "dest_user"],
                "target_process": ["target_process", "destination_process"],
                "target_pid": ["target_pid", "destination_pid"],
                "network_src_ip": ["src_ip", "source_ip", "source_address"],
                "network_dst_ip": ["dst_ip", "dest_ip", "destination_ip", "destination_address"],
                "network_dst_port": ["dst_port", "dest_port", "destination_port"],
                "file_path": ["file", "filename", "file_path", "path"],
                "registry_key": ["registry_key", "reg_key", "key"],
                "command_line": ["command_line", "cmdline", "command"],
                "parent_process": ["parent_process", "parent_image"],
                "parent_pid": ["parent_pid", "ppid"]
            }
        }
    
    def detect_log_format(self, event_dict: Dict[str, Any]) -> str:
        """
        Auto-detect log format based on event structure.
        
        Parameters
        ----------
        event_dict : dict
            Single log event as dictionary
            
        Returns
        -------
        str
            Detected format: 'windows_event', 'sysmon', 'linux_audit', or 'custom_json'
        """
        try:
            # Check for Windows Event Log indicators
            if any(key in event_dict for key in ['EventID', 'Computer', 'TimeCreated']):
                # Check if it's Sysmon (has specific Sysmon fields)
                if any(key in event_dict for key in ['Image', 'UtcTime', 'ProcessGuid']):
                    return 'sysmon'
                return 'windows_event'
            
            # Check for Linux Audit indicators
            if any(key in event_dict for key in ['type', 'msg', 'audit']):
                return 'linux_audit'
            
            # Default to custom JSON
            return 'custom_json'
            
        except Exception as e:
            logger.error(f"Error detecting log format: {e}")
            return 'custom_json'
    
    def parse_windows_event(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Windows Event Log format.
        
        Parameters
        ----------
        event_dict : dict
            Raw Windows event
            
        Returns
        -------
        dict
            Normalized event
        """
        try:
            normalized = self.normalized_schema.copy()
            mappings = self.field_mappings.get('windows_event', {})
            
            # Extract fields using mappings
            for norm_field, possible_fields in mappings.items():
                for field in possible_fields:
                    if field in event_dict:
                        normalized[norm_field] = event_dict[field]
                        break
            
            # Determine event type based on EventID
            event_id = normalized.get('event_id')
            if event_id:
                normalized['event_type'] = self.map_windows_event_type(event_id)
            
            # Store raw event
            normalized['raw_event'] = event_dict
            
            self.stats['windows_events'] += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error parsing Windows event: {e}")
            self.stats['parsing_errors'] += 1
            return None
    
    def parse_sysmon(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Sysmon event format.
        
        Parameters
        ----------
        event_dict : dict
            Raw Sysmon event
            
        Returns
        -------
        dict
            Normalized event
        """
        try:
            normalized = self.normalized_schema.copy()
            mappings = self.field_mappings.get('sysmon', {})
            
            # Extract fields using mappings
            for norm_field, possible_fields in mappings.items():
                for field in possible_fields:
                    if field in event_dict:
                        normalized[norm_field] = event_dict[field]
                        break
            
            # Determine event type based on Sysmon EventID
            event_id = normalized.get('event_id')
            if event_id:
                normalized['event_type'] = self.map_sysmon_event_type(event_id)
            
            # Store raw event
            normalized['raw_event'] = event_dict
            
            self.stats['sysmon_events'] += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error parsing Sysmon event: {e}")
            self.stats['parsing_errors'] += 1
            return None
    
    def parse_linux_audit(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Linux Audit log format.
        
        Parameters
        ----------
        event_dict : dict
            Raw Linux audit event
            
        Returns
        -------
        dict
            Normalized event
        """
        try:
            normalized = self.normalized_schema.copy()
            mappings = self.field_mappings.get('linux_audit', {})
            
            # Extract fields using mappings
            for norm_field, possible_fields in mappings.items():
                for field in possible_fields:
                    if field in event_dict:
                        normalized[norm_field] = event_dict[field]
                        break
            
            # Map audit event types
            audit_type = event_dict.get('type', '')
            normalized['event_type'] = self.map_linux_audit_event_type(audit_type)
            
            # Store raw event
            normalized['raw_event'] = event_dict
            
            self.stats['linux_audit_events'] += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error parsing Linux audit event: {e}")
            self.stats['parsing_errors'] += 1
            return None
    
    def parse_custom_json(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse custom JSON format with flexible field mapping.
        
        Parameters
        ----------
        event_dict : dict
            Raw custom JSON event
            
        Returns
        -------
        dict
            Normalized event
        """
        try:
            normalized = self.normalized_schema.copy()
            mappings = self.field_mappings.get('custom_json', {})
            
            # Extract fields using mappings
            for norm_field, possible_fields in mappings.items():
                for field in possible_fields:
                    if field in event_dict:
                        normalized[norm_field] = event_dict[field]
                        break
            
            # Try to infer event type from available fields
            if not normalized['event_type']:
                normalized['event_type'] = self.infer_event_type(event_dict)
            
            # Store raw event
            normalized['raw_event'] = event_dict
            
            self.stats['custom_json_events'] += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error parsing custom JSON event: {e}")
            self.stats['parsing_errors'] += 1
            return None
    
    def normalize_event(self, event_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize event to standard schema.
        
        Parameters
        ----------
        event_dict : dict
            Raw event dictionary
            
        Returns
        -------
        dict or None
            Normalized event or None if parsing failed
        """
        try:
            self.stats['total_events_processed'] += 1
            
            # Detect format
            log_format = self.detect_log_format(event_dict)
            
            # Parse based on format
            if log_format == 'windows_event':
                normalized = self.parse_windows_event(event_dict)
            elif log_format == 'sysmon':
                normalized = self.parse_sysmon(event_dict)
            elif log_format == 'linux_audit':
                normalized = self.parse_linux_audit(event_dict)
            else:  # custom_json
                normalized = self.parse_custom_json(event_dict)
            
            if normalized:
                # Post-process normalization
                normalized = self.post_process_event(normalized)
                self.stats['successful_normalizations'] += 1
                return normalized
            else:
                self.stats['normalization_errors'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error normalizing event: {e}")
            self.stats['normalization_errors'] += 1
            return None
    
    def post_process_event(self, normalized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process normalized event for consistency.
        
        Parameters
        ----------
        normalized : dict
            Normalized event
            
        Returns
        -------
        dict
            Post-processed event
        """
        try:
            # Normalize timestamp to ISO format
            if normalized['timestamp']:
                normalized['timestamp'] = self.normalize_timestamp(normalized['timestamp'])
            
            # Clean and normalize usernames
            if normalized['source_user']:
                normalized['source_user'] = self.normalize_username(normalized['source_user'])
            if normalized['target_user']:
                normalized['target_user'] = self.normalize_username(normalized['target_user'])
            
            # Normalize hostnames
            if normalized['source_host']:
                normalized['source_host'] = self.normalize_hostname(normalized['source_host'])
            if normalized['target_host']:
                normalized['target_host'] = self.normalize_hostname(normalized['target_host'])
            
            # Normalize process names
            if normalized['source_process']:
                normalized['source_process'] = self.normalize_process_name(normalized['source_process'])
            if normalized['target_process']:
                normalized['target_process'] = self.normalize_process_name(normalized['target_process'])
            if normalized['parent_process']:
                normalized['parent_process'] = self.normalize_process_name(normalized['parent_process'])
            
            # Convert PIDs to integers
            for pid_field in ['source_pid', 'target_pid', 'parent_pid']:
                if normalized[pid_field]:
                    try:
                        normalized[pid_field] = int(normalized[pid_field])
                    except (ValueError, TypeError):
                        normalized[pid_field] = None
            
            # Convert port to integer
            if normalized['network_dst_port']:
                try:
                    normalized['network_dst_port'] = int(normalized['network_dst_port'])
                except (ValueError, TypeError):
                    normalized['network_dst_port'] = None
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error post-processing event: {e}")
            return normalized
    
    def normalize_timestamp(self, timestamp: Union[str, int, float]) -> str:
        """Normalize timestamp to ISO 8601 format."""
        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(timestamp)
                return dt.isoformat()
            elif isinstance(timestamp, str):
                # Try common timestamp formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S'
                ]
                
                for fmt in formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
                
                # If no format matches, return as-is
                return timestamp
            else:
                return str(timestamp)
        except Exception:
            return str(timestamp)
    
    def normalize_username(self, username: str) -> str:
        """Normalize username format."""
        if not username:
            return username
        
        # Convert to lowercase for consistency
        username = username.lower().strip()
        
        # Handle different domain formats
        if '\\' in username:
            # DOMAIN\user format
            return username
        elif '@' in username:
            # user@domain format - convert to DOMAIN\user
            user, domain = username.split('@', 1)
            return f"{domain}\\{user}"
        else:
            # Just username - assume local
            return username
    
    def normalize_hostname(self, hostname: str) -> str:
        """Normalize hostname format."""
        if not hostname:
            return hostname
        
        # Convert to uppercase for consistency
        hostname = hostname.upper().strip()
        
        # Remove domain suffix if present
        if '.' in hostname:
            hostname = hostname.split('.')[0]
        
        return hostname
    
    def normalize_process_name(self, process: str) -> str:
        """Normalize process name."""
        if not process:
            return process
        
        # Extract just the filename from full path
        if '\\' in process:
            process = process.split('\\')[-1]
        elif '/' in process:
            process = process.split('/')[-1]
        
        # Convert to lowercase for consistency
        return process.lower().strip()
    
    def map_windows_event_type(self, event_id: int) -> str:
        """Map Windows Event ID to event type."""
        event_type_map = {
            4624: 'user_logon',
            4625: 'user_logon_failed',
            4634: 'user_logoff',
            4648: 'explicit_logon',
            4688: 'process_create',
            4689: 'process_terminate',
            4697: 'service_install',
            4698: 'scheduled_task_create',
            4699: 'scheduled_task_delete',
            4700: 'scheduled_task_enable',
            4701: 'scheduled_task_disable',
            4702: 'scheduled_task_update',
            4720: 'user_account_create',
            4722: 'user_account_enable',
            4724: 'user_account_reset_password',
            4725: 'user_account_disable',
            4726: 'user_account_delete',
            4728: 'user_added_to_group',
            4732: 'user_added_to_local_group',
            4733: 'user_removed_from_local_group',
            4738: 'user_account_change',
            4740: 'user_account_lockout',
            4767: 'user_account_unlock',
            5140: 'network_share_access',
            5156: 'network_connection_allow',
            5157: 'network_connection_block'
        }
        
        return event_type_map.get(event_id, f'windows_event_{event_id}')
    
    def map_sysmon_event_type(self, event_id: int) -> str:
        """Map Sysmon Event ID to event type."""
        event_type_map = {
            1: 'process_create',
            2: 'file_create_time_change',
            3: 'network_connection',
            4: 'sysmon_service_state_change',
            5: 'process_terminate',
            6: 'driver_load',
            7: 'image_load',
            8: 'create_remote_thread',
            9: 'raw_access_read',
            10: 'process_access',
            11: 'file_create',
            12: 'registry_event_create_delete',
            13: 'registry_event_value_set',
            14: 'registry_event_key_value_rename',
            15: 'file_create_stream_hash',
            16: 'sysmon_config_change',
            17: 'pipe_event_create',
            18: 'pipe_event_connect',
            19: 'wmi_event_filter',
            20: 'wmi_event_consumer',
            21: 'wmi_event_consumer_to_filter',
            22: 'dns_event',
            23: 'file_delete',
            24: 'clipboard_capture',
            25: 'process_tampering'
        }
        
        return event_type_map.get(event_id, f'sysmon_event_{event_id}')
    
    def map_linux_audit_event_type(self, audit_type: str) -> str:
        """Map Linux audit type to event type."""
        type_map = {
            'SYSCALL': 'system_call',
            'EXECVE': 'process_execute',
            'PATH': 'file_access',
            'CWD': 'working_directory_change',
            'USER_LOGIN': 'user_login',
            'USER_LOGOUT': 'user_logout',
            'USER_START': 'user_session_start',
            'USER_END': 'user_session_end',
            'CRED_ACQ': 'credential_acquisition',
            'CRED_DISP': 'credential_disposal',
            'USER_ACCT': 'user_account_change',
            'ADD_USER': 'user_add',
            'DEL_USER': 'user_delete',
            'ADD_GROUP': 'group_add',
            'DEL_GROUP': 'group_delete',
            'CHGRP_ID': 'group_change'
        }
        
        return type_map.get(audit_type, f'audit_{audit_type.lower()}')
    
    def infer_event_type(self, event_dict: Dict[str, Any]) -> str:
        """Infer event type from custom JSON fields."""
        # Check for process-related fields
        if any(field in event_dict for field in ['process', 'command_line', 'pid']):
            return 'process_activity'
        
        # Check for network-related fields
        if any(field in event_dict for field in ['src_ip', 'dst_ip', 'port', 'protocol']):
            return 'network_activity'
        
        # Check for file-related fields
        if any(field in event_dict for field in ['file', 'filename', 'path']):
            return 'file_activity'
        
        # Check for authentication-related fields
        if any(field in event_dict for field in ['login', 'logon', 'authentication']):
            return 'authentication'
        
        # Default
        return 'unknown'
    
    def parse_log_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse entire log file and return normalized events.
        
        Parameters
        ----------
        file_path : str
            Path to log file (JSON, JSONL, or text)
            
        Returns
        -------
        list
            List of normalized events
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return []
            
            logger.info(f"Parsing log file: {file_path}")
            
            normalized_events = []
            
            if file_path.suffix.lower() == '.json':
                # JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        # Array of events
                        for event in data:
                            normalized = self.normalize_event(event)
                            if normalized:
                                normalized_events.append(normalized)
                    else:
                        # Single event
                        normalized = self.normalize_event(data)
                        if normalized:
                            normalized_events.append(normalized)
            
            elif file_path.suffix.lower() == '.jsonl':
                # JSONL file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            event = json.loads(line)
                            normalized = self.normalize_event(event)
                            if normalized:
                                normalized_events.append(normalized)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error at line {line_num}: {e}")
                            continue
            
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return []
            
            logger.info(f"Successfully parsed {len(normalized_events)} events from {file_path}")
            return normalized_events
            
        except Exception as e:
            logger.error(f"Error parsing log file {file_path}: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset parsing statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def save_normalized_events(self, events: List[Dict[str, Any]], output_path: str):
        """
        Save normalized events to JSON file.
        
        Parameters
        ----------
        events : list
            List of normalized events
        output_path : str
            Output file path
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, default=str)
            
            logger.info(f"Saved {len(events)} normalized events to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving normalized events: {e}")


def main():
    """Example usage of UniversalLogParser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Log Parser for Security Events')
    parser.add_argument('input_file', help='Input log file (JSON/JSONL)')
    parser.add_argument('-o', '--output', help='Output file for normalized events')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize parser
    parser = UniversalLogParser(config_path=args.config)
    
    # Parse log file
    normalized_events = parser.parse_log_file(args.input_file)
    
    # Print statistics
    stats = parser.get_statistics()
    print("\n=== Parsing Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save results if output specified
    if args.output and normalized_events:
        parser.save_normalized_events(normalized_events, args.output)
    
    # Print sample events
    if normalized_events:
        print(f"\n=== Sample Normalized Events (first 3) ===")
        for i, event in enumerate(normalized_events[:3]):
            print(f"\nEvent {i+1}:")
            for key, value in event.items():
                if key != 'raw_event' and value is not None:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()