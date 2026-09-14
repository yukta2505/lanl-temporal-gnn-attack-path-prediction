#!/usr/bin/env python3
"""
JSONL to JSON Converter - Professional Implementation
Converts JSONL (JSON Lines) files to standard JSON format for THGNN processing.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - PROFESSIONAL CONVERTER
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JSONLConverter:
    """Professional JSONL to JSON converter with comprehensive error handling"""
    
    def __init__(self):
        self.stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_lines': 0,
            'processing_time': 0,
            'file_size_input': 0,
            'file_size_output': 0
        }
    
    def validate_input_file(self, input_path: Path) -> Tuple[bool, str]:
        """Validate input JSONL file"""
        try:
            if not input_path.exists():
                return False, f"Input file does not exist: {input_path}"
            
            if not input_path.is_file():
                return False, f"Input path is not a file: {input_path}"
            
            if input_path.suffix.lower() not in ['.jsonl', '.ndjson', '.json']:
                return False, f"Unsupported file extension: {input_path.suffix}"
            
            if input_path.stat().st_size == 0:
                return False, "Input file is empty"
            
            if input_path.stat().st_size > 1024 * 1024 * 1024:  # 1GB limit
                return False, "Input file too large (max 1GB)"
            
            return True, "Input file validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def read_jsonl_file(self, input_path: Path) -> Tuple[bool, List[Dict[Any, Any]], str]:
        """Read and parse JSONL file"""
        try:
            logger.info(f"Reading JSONL file: {input_path}")
            
            self.stats['file_size_input'] = input_path.stat().st_size
            events = []
            line_number = 0
            
            with open(input_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line_number += 1
                    self.stats['total_lines'] = line_number
                    
                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON line
                        event = json.loads(line)
                        events.append(event)
                        self.stats['valid_lines'] += 1
                        
                        # Progress logging for large files
                        if line_number % 10000 == 0:
                            logger.info(f"Processed {line_number:,} lines, {len(events):,} valid events")
                    
                    except json.JSONDecodeError as e:
                        self.stats['invalid_lines'] += 1
                        logger.warning(f"Invalid JSON on line {line_number}: {str(e)}")
                        logger.debug(f"Invalid line content: {line[:100]}...")
                        continue
                    
                    except Exception as e:
                        self.stats['invalid_lines'] += 1
                        logger.error(f"Unexpected error on line {line_number}: {str(e)}")
                        continue
            
            if not events:
                return False, [], "No valid JSON events found in file"
            
            logger.info(f"Successfully parsed {len(events):,} events from {line_number:,} lines")
            return True, events, f"Successfully read {len(events)} events"
            
        except FileNotFoundError:
            return False, [], f"File not found: {input_path}"
        except PermissionError:
            return False, [], f"Permission denied reading file: {input_path}"
        except UnicodeDecodeError as e:
            return False, [], f"Unicode decode error: {str(e)}"
        except Exception as e:
            return False, [], f"Unexpected error reading file: {str(e)}"
    
    def write_json_file(self, events: List[Dict[Any, Any]], output_path: Path, indent: int = 2) -> Tuple[bool, str]:
        """Write events to JSON file"""
        try:
            logger.info(f"Writing JSON file: {output_path}")
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(events, file, indent=indent, ensure_ascii=False, separators=(',', ': '))
            
            self.stats['file_size_output'] = output_path.stat().st_size
            
            logger.info(f"Successfully wrote {len(events):,} events to {output_path}")
            return True, f"Successfully wrote {len(events)} events"
            
        except PermissionError:
            return False, f"Permission denied writing to: {output_path}"
        except OSError as e:
            return False, f"OS error writing file: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error writing file: {str(e)}"
    
    def convert_file(self, input_path: Path, output_path: Path = None, indent: int = 2) -> Tuple[bool, str]:
        """Convert JSONL file to JSON format"""
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, validation_msg = self.validate_input_file(input_path)
            if not is_valid:
                return False, validation_msg
            
            # Generate output path if not provided
            if output_path is None:
                output_path = input_path.with_suffix('.json')
                if output_path == input_path:
                    output_path = input_path.with_name(f"{input_path.stem}_converted.json")
            
            # Read JSONL file
            success, events, read_msg = self.read_jsonl_file(input_path)
            if not success:
                return False, read_msg
            
            # Write JSON file
            success, write_msg = self.write_json_file(events, output_path, indent)
            if not success:
                return False, write_msg
            
            # Calculate processing time
            self.stats['processing_time'] = time.time() - start_time
            
            # Generate success message
            success_msg = f"""
Conversion completed successfully!
Input:  {input_path} ({self.stats['file_size_input']:,} bytes)
Output: {output_path} ({self.stats['file_size_output']:,} bytes)
Events: {len(events):,} valid events processed
Time:   {self.stats['processing_time']:.2f} seconds
            """.strip()
            
            return True, success_msg
            
        except Exception as e:
            return False, f"Conversion failed: {str(e)}"
    
    def print_statistics(self):
        """Print detailed conversion statistics"""
        print("\n" + "="*60)
        print("CONVERSION STATISTICS")
        print("="*60)
        print(f"Total lines processed:    {self.stats['total_lines']:,}")
        print(f"Valid JSON lines:         {self.stats['valid_lines']:,}")
        print(f"Invalid lines skipped:    {self.stats['invalid_lines']:,}")
        print(f"Success rate:             {(self.stats['valid_lines']/max(1,self.stats['total_lines'])*100):.1f}%")
        print(f"Input file size:          {self.stats['file_size_input']:,} bytes")
        print(f"Output file size:         {self.stats['file_size_output']:,} bytes")
        print(f"Processing time:          {self.stats['processing_time']:.2f} seconds")
        if self.stats['processing_time'] > 0:
            print(f"Processing rate:          {self.stats['valid_lines']/self.stats['processing_time']:.0f} events/second")
        print("="*60)

def main():
    """Main converter function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Convert JSONL (JSON Lines) files to standard JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jsonl_to_json_converter.py input.jsonl
  python jsonl_to_json_converter.py input.jsonl -o output.json
  python jsonl_to_json_converter.py input.jsonl --indent 4 --verbose
  python jsonl_to_json_converter.py *.jsonl --batch
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input JSONL file(s) to convert'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output JSON file path (default: input_file.json)'
    )
    
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation level (default: 2)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple files in batch mode'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed conversion statistics'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process files
    converter = JSONLConverter()
    total_files = len(args.input_files)
    successful_conversions = 0
    
    print(f"Starting JSONL to JSON conversion for {total_files} file(s)...")
    
    for i, input_file in enumerate(args.input_files, 1):
        input_path = Path(input_file)
        
        print(f"\n[{i}/{total_files}] Processing: {input_path}")
        
        # Determine output path
        if args.output and not args.batch:
            output_path = Path(args.output)
        else:
            output_path = None
        
        # Convert file
        success, message = converter.convert_file(input_path, output_path, args.indent)
        
        if success:
            successful_conversions += 1
            print(f"[DONE] SUCCESS: {message}")
        else:
            print(f"[ERROR] FAILED: {message}")
        
        # Show statistics if requested
        if args.stats:
            converter.print_statistics()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:              {total_files}")
    print(f"Successful conversions:   {successful_conversions}")
    print(f"Failed conversions:       {total_files - successful_conversions}")
    print(f"Success rate:             {(successful_conversions/total_files*100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if successful_conversions == total_files else 1)

if __name__ == '__main__':
    main()