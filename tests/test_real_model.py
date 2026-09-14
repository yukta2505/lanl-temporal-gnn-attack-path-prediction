"""Test loading the REAL trained model"""
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'app'))

from real_ml_processor import RealMLProcessor

print("="*80)
print("TESTING REAL MODEL LOADING")
print("="*80)

try:
    processor = RealMLProcessor()
    print("\n✅ SUCCESS! Real model loaded")
    print(f"   Model type: {type(processor.model).__name__}")
    print(f"   Device: {processor.device}")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
