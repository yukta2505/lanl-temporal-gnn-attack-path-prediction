#!/usr/bin/env python3
"""
Run all THGNN tests from the tests directory
"""
import sys
import subprocess
from pathlib import Path

# Get the tests directory
tests_dir = Path(__file__).parent

# Define all test files
test_files = [
    'test_full_power.py',
    'test_perfect_match.py',
    'test_real_model.py',
    'test_full_system.py',
    'FINAL_COMPLETE_TEST.py'
]

print("="*80)
print("RUNNING ALL THGNN TESTS")
print("="*80)

results = {}

for test_file in test_files:
    test_path = tests_dir / test_file
    
    if not test_path.exists():
        print(f"\n❌ {test_file} - NOT FOUND")
        results[test_file] = "NOT FOUND"
        continue
    
    print(f"\n{'='*80}")
    print(f"Running: {test_file}")
    print('='*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(tests_dir.parent),  # Run from project root
            capture_output=False,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"\n✅ {test_file} - PASSED")
            results[test_file] = "PASSED"
        else:
            print(f"\n❌ {test_file} - FAILED (exit code: {result.returncode})")
            results[test_file] = "FAILED"
    
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  {test_file} - TIMEOUT")
        results[test_file] = "TIMEOUT"
    
    except Exception as e:
        print(f"\n❌ {test_file} - ERROR: {e}")
        results[test_file] = "ERROR"

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

passed = sum(1 for status in results.values() if status == "PASSED")
failed = sum(1 for status in results.values() if status in ["FAILED", "ERROR", "TIMEOUT", "NOT FOUND"])

for test_file, status in results.items():
    icon = "✅" if status == "PASSED" else "❌"
    print(f"{icon} {test_file:40s} {status}")

print("\n" + "="*80)
print(f"TOTAL: {passed}/{len(test_files)} PASSED")
print("="*80)

if passed == len(test_files):
    print("\n🎉 ALL TESTS PASSED! SYSTEM FULLY FUNCTIONAL! 🎉")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed} test(s) failed")
    sys.exit(1)
