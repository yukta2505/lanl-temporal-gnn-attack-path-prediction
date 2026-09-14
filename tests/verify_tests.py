#!/usr/bin/env python3
"""
Quick verification script to ensure all tests are properly configured
"""
import sys
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent

print("="*80)
print("THGNN TEST VERIFICATION")
print("="*80)

# Check project structure
print("\n1. Checking project structure...")
required_dirs = ['src', 'app', 'tests', 'checkpoints', 'config']
for dir_name in required_dirs:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"   ✅ {dir_name}/ exists")
    else:
        print(f"   ❌ {dir_name}/ missing")

# Check test files
print("\n2. Checking test files...")
test_files = [
    'test_perfect_match.py',
    'test_full_power.py',
    'test_real_model.py',
    'test_full_system.py',
    'FINAL_COMPLETE_TEST.py',
    'run_all_tests.py',
    'README.md'
]

tests_dir = project_root / 'tests'
for test_file in test_files:
    test_path = tests_dir / test_file
    if test_path.exists():
        print(f"   ✅ {test_file}")
    else:
        print(f"   ❌ {test_file} missing")

# Check critical files
print("\n3. Checking critical files...")
critical_files = [
    'checkpoints/REAL_TRAINED_MODEL_AUC0.8545.pth',
    'src/models/perfect_legacy_thgnn.py',
    'app/real_ml_processor.py',
    'requirements.txt'
]

for file_path in critical_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ⚠️  {file_path} missing (may be optional)")

# Check imports
print("\n4. Checking Python imports...")
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'app'))

try:
    from models.perfect_legacy_thgnn import PerfectLegacyTHGNN
    print("   ✅ models.perfect_legacy_thgnn")
except ImportError as e:
    print(f"   ❌ models.perfect_legacy_thgnn: {e}")

try:
    from real_ml_processor import RealMLProcessor
    print("   ✅ real_ml_processor")
except ImportError as e:
    print(f"   ❌ real_ml_processor: {e}")

try:
    from data_processing.graph_builder import TemporalGraphBuilder
    print("   ✅ data_processing.graph_builder")
except ImportError as e:
    print(f"   ❌ data_processing.graph_builder: {e}")

try:
    import torch
    print(f"   ✅ torch (version {torch.__version__})")
except ImportError:
    print("   ❌ torch not installed")

# Summary
print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nTo run tests:")
print("  python tests/FINAL_COMPLETE_TEST.py")
print("  python tests/run_all_tests.py")
print("  pytest tests/")
print("\n" + "="*80)
