#!/usr/bin/env python3
"""
Critical Fix: Numpy Compatibility Issues
Handles numpy version compatibility problems, especially numpy._core module errors.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - CRITICAL NUMPY FIX
"""

import sys
import logging
import warnings
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_numpy_compatibility():
    """
    Fix numpy compatibility issues, especially numpy._core module problems.
    This is a critical fix for the 'No module named numpy._core' error.
    """
    try:
        import numpy as np
        
        # Check numpy version
        numpy_version = np.__version__
        logger.info(f"Numpy version: {numpy_version}")
        
        # Handle numpy._core compatibility
        try:
            # Try to import numpy._core (newer versions)
            import numpy._core
            logger.info(" numpy._core available")
        except ImportError:
            # Fallback for older numpy versions
            logger.warning("numpy._core not available, using compatibility layer")
            
            # Create compatibility layer
            import numpy.core as _core
            sys.modules['numpy._core'] = _core
            
            # Also handle specific submodules that might be missing
            try:
                import numpy.core.multiarray as multiarray
                sys.modules['numpy._core.multiarray'] = multiarray
            except ImportError:
                pass
            
            try:
                import numpy.core.umath as umath
                sys.modules['numpy._core.umath'] = umath
            except ImportError:
                pass
        
        # Suppress numpy warnings that can cause issues
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        warnings.filterwarnings('ignore', category=np.ComplexWarning)
        
        # Set numpy error handling to be more forgiving
        np.seterr(all='ignore')
        
        logger.info(" Numpy compatibility fixes applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to fix numpy compatibility: {e}")
        return False

def safe_numpy_import() -> Optional[Any]:
    """
    Safely import numpy with compatibility fixes.
    
    Returns
    -------
    Optional[Any]
        Numpy module or None if import fails
    """
    try:
        # Apply compatibility fixes first
        fix_numpy_compatibility()
        
        # Import numpy
        import numpy as np
        
        # Verify basic functionality
        test_array = np.array([1, 2, 3])
        assert test_array.shape == (3,)
        
        logger.info(" Numpy imported successfully with compatibility fixes")
        return np
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to import numpy: {e}")
        return None

def fix_torch_numpy_compatibility():
    """
    Fix PyTorch-Numpy compatibility issues.
    """
    try:
        import torch
        import numpy as np
        
        # Ensure torch and numpy are compatible
        torch_version = torch.__version__
        numpy_version = np.__version__
        
        logger.info(f"PyTorch version: {torch_version}")
        logger.info(f"Numpy version: {numpy_version}")
        
        # Test basic interoperability
        torch_tensor = torch.tensor([1, 2, 3])
        numpy_array = torch_tensor.numpy()
        back_to_torch = torch.from_numpy(numpy_array)
        
        assert torch.equal(torch_tensor, back_to_torch)
        
        logger.info(" PyTorch-Numpy compatibility verified")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] PyTorch-Numpy compatibility issue: {e}")
        return False

def apply_all_compatibility_fixes():
    """
    Apply all compatibility fixes for the THGNN system.
    """
    logger.info("Applying all compatibility fixes...")
    
    success = True
    
    # Fix numpy compatibility
    if not fix_numpy_compatibility():
        logger.error("[ERROR] Numpy compatibility fix failed")
        success = False
    
    # Fix torch-numpy compatibility
    if not fix_torch_numpy_compatibility():
        logger.error("[ERROR] PyTorch-Numpy compatibility fix failed")
        success = False
    
    if success:
        logger.info(" All compatibility fixes applied successfully")
    else:
        logger.error("[ERROR] Some compatibility fixes failed")
    
    return success

# Apply fixes on import
if __name__ != '__main__':
    apply_all_compatibility_fixes()

if __name__ == '__main__':
    """Test compatibility fixes"""
    logger.info("Testing compatibility fixes...")
    
    success = apply_all_compatibility_fixes()
    
    if success:
        logger.info(" All compatibility fixes working!")
    else:
        logger.error("[ERROR] Compatibility fixes need attention")
        sys.exit(1)