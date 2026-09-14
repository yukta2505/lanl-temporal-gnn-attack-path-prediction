#!/usr/bin/env python3

import os
import sys
import requests
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and verify Mordor security datasets."""
    
    def __init__(self, output_dir: str = "data/raw/mordor_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {
            "empire_psexec_dcerpc": {
                "url": "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/lateral_movement/host/empire_psexec_dcerpc_tcp_svcctl.zip",
                "description": "Empire PsExec lateral movement via DCERPC",
                "size_mb": 15
            },
            "covenant_wmi_execution": {
                "url": "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/covenant_wmi_wmiprv_process_execution.zip", 
                "description": "Covenant WMI process execution",
                "size_mb": 8
            },
            "apt29_day1_manual": {
                "url": "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/campaign/apt29/day1/apt29_evals_day1_manual.zip",
                "description": "APT29 Day 1 evaluation campaign",
                "size_mb": 45
            },
            "empire_invoke_wmi": {
                "url": "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/lateral_movement/host/empire_invoke_wmi.zip",
                "description": "Empire Invoke-WMI lateral movement", 
                "size_mb": 12
            },
            "empire_powershell_dcerpc": {
                "url": "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/empire_powershell_dcerpc_wmi_process_execution.zip",
                "description": "Empire PowerShell DCERPC WMI execution",
                "size_mb": 18
            }
        }
    
    def download_file(self, url: str, filepath: Path, expected_size_mb: Optional[int] = None) -> bool:
        """Download file with progress bar and basic validation."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded {filepath.name}: {file_size_mb:.1f} MB")
            
            if expected_size_mb and abs(file_size_mb - expected_size_mb) > expected_size_mb * 0.5:
                logger.warning(f"File size mismatch for {filepath.name}: expected ~{expected_size_mb}MB, got {file_size_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def verify_zip(self, filepath: Path) -> bool:
        """Verify ZIP file integrity."""
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.testzip()
            logger.info(f"ZIP verification passed: {filepath.name}")
            return True
        except Exception as e:
            logger.error(f"ZIP verification failed for {filepath.name}: {e}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download and verify a single dataset."""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        url = dataset_info["url"]
        filepath = self.output_dir / f"{dataset_name}.zip"
        
        if filepath.exists():
            logger.info(f"Dataset already exists: {filepath.name}")
            if self.verify_zip(filepath):
                return True
            else:
                logger.info(f"Re-downloading corrupted file: {filepath.name}")
                filepath.unlink()
        
        logger.info(f"Downloading {dataset_info['description']}...")
        
        if self.download_file(url, filepath, dataset_info["size_mb"]):
            if self.verify_zip(filepath):
                return True
            else:
                filepath.unlink()
                return False
        
        return False
    
    def download_all(self) -> Dict[str, bool]:
        """Download all datasets."""
        results = {}
        
        logger.info(f"Starting download of {len(self.datasets)} datasets...")
        
        for dataset_name in self.datasets:
            logger.info(f"\n--- Downloading {dataset_name} ---")
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results
    
    def generate_manifest(self, results: Dict[str, bool]) -> None:
        """Generate download manifest."""
        manifest_path = self.output_dir / "download_manifest.txt"
        
        with open(manifest_path, 'w') as f:
            f.write("THGNN Dataset Download Manifest\n")
            f.write("=" * 40 + "\n\n")
            
            successful = sum(results.values())
            total = len(results)
            f.write(f"Download Summary: {successful}/{total} successful\n\n")
            
            for dataset_name, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                dataset_info = self.datasets[dataset_name]
                f.write(f"{dataset_name}: {status}\n")
                f.write(f"  Description: {dataset_info['description']}\n")
                f.write(f"  URL: {dataset_info['url']}\n")
                
                if success:
                    filepath = self.output_dir / f"{dataset_name}.zip"
                    if filepath.exists():
                        size_mb = filepath.stat().st_size / (1024 * 1024)
                        f.write(f"  Size: {size_mb:.1f} MB\n")
                
                f.write("\n")
        
        logger.info(f"Manifest saved to: {manifest_path}")

def main():
    """Main execution function."""
    downloader = DatasetDownloader()
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        if dataset_name == "--list":
            print("Available datasets:")
            for name, info in downloader.datasets.items():
                print(f"  {name}: {info['description']}")
            return
        
        success = downloader.download_dataset(dataset_name)
        sys.exit(0 if success else 1)
    else:
        results = downloader.download_all()
        downloader.generate_manifest(results)
        
        successful = sum(results.values())
        total = len(results)
        
        if successful == total:
            logger.info(f"All {total} datasets downloaded successfully!")
            sys.exit(0)
        else:
            logger.error(f"Only {successful}/{total} datasets downloaded successfully")
            sys.exit(1)

if __name__ == "__main__":
    main()
