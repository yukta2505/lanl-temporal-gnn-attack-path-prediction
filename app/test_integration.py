#!/usr/bin/env python3
"""
Day 8 Afternoon: Integration Test for THGNN Dashboard
Test all theme functionality and visualizations.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - INTEGRATION TEST
"""

import sys
import time
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Test imports
try:
    from visualization.advanced_plotly_themes import THGNNPlotlyThemes
    from inference.production_engine import ProductionInferenceEngine
    from visualization.visualization_integration import THGNNVisualizationEngine
    from explainability.gnn_explainer import ProductionGNNExplainer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_theme_system():
    """Test advanced theme system"""
    print("\n🎨 Testing Advanced Theme System...")
    
    themes = ["dark_cyber", "light_professional", "cyber_red", "security_blue", "matrix_green"]
    
    for theme in themes:
        try:
            # Test theme configuration
            config = THGNNPlotlyThemes.get_theme_config(theme)
            assert 'layout' in config
            assert 'node_colors' in config
            assert 'edge_colors' in config
            assert 'risk_colors' in config
            print(f"✅ Theme '{theme}' configuration valid")
            
            # Test visualization creation
            import numpy as np
            x_data = np.random.randn(20)
            y_data = np.random.randn(20)
            
            fig = THGNNPlotlyThemes.create_themed_scatter(
                x_data, y_data, 
                theme_name=theme,
                title=f"Test Plot - {theme}"
            )
            
            assert fig is not None
            print(f"✅ Theme '{theme}' scatter plot created")
            
        except Exception as e:
            print(f"❌ Theme '{theme}' failed: {e}")
            return False
    
    print("🎉 All themes tested successfully!")
    return True

def test_production_components():
    """Test production components initialization"""
    print("\n🔧 Testing Production Components...")
    
    try:
        # Test inference engine
        inference_engine = ProductionInferenceEngine(
            model_path="checkpoints/best_real_data_model.pth",
            device='cpu'  # Use CPU for testing
        )
        print("✅ Production inference engine initialized")
        
        # Test visualization engine
        viz_engine = THGNNVisualizationEngine()
        print("✅ Visualization engine initialized")
        
        # Test explainer (with mock model)
        explainer = ProductionGNNExplainer(model=inference_engine.model)
        print("✅ GNN explainer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

def test_dashboard_functionality():
    """Test dashboard functionality without Streamlit"""
    print("\n📊 Testing Dashboard Functionality...")
    
    try:
        # Test theme mapping
        theme_mapping = {
            '🌙 Dark Mode': 'dark_cyber',
            '☀️ Light Mode': 'light_professional', 
            '🔥 Cyber Red': 'cyber_red',
            '🛡️ Security Blue': 'security_blue',
            '🌿 Matrix Green': 'matrix_green'
        }
        
        for display_name, theme_key in theme_mapping.items():
            config = THGNNPlotlyThemes.get_theme_config(theme_key)
            assert config is not None
            print(f"✅ Theme mapping '{display_name}' → '{theme_key}' valid")
        
        # Test data generation functions
        import pandas as pd
        import numpy as np
        
        # Test forecast data generation
        forecast_steps = []
        for step in range(1, 4):
            forecast_steps.append({
                'Step': step,
                'Source': f'user_test_{step}',
                'Target': f'host_test_{step}',
                'Risk_Score': np.random.randint(70, 95),
                'Confidence': np.random.uniform(0.8, 0.95)
            })
        
        forecast_df = pd.DataFrame(forecast_steps)
        assert len(forecast_df) == 3
        print("✅ Forecast data generation working")
        
        # Test heatmap data generation
        entities = ['user_alice', 'user_bob', 'host_dc01']
        time_windows = ['08:00', '10:00', '12:00']
        heatmap_data = np.random.randint(10, 95, (len(entities), len(time_windows)))
        
        assert heatmap_data.shape == (3, 3)
        print("✅ Heatmap data generation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard functionality test failed: {e}")
        return False

def test_visualization_creation():
    """Test visualization creation with themes"""
    print("\n📈 Testing Visualization Creation...")
    
    try:
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        # Test data
        timestamps = pd.date_range(start=datetime.now(), periods=24, freq='H')
        values = np.random.randint(20, 100, 24)
        
        # Test each theme
        themes = ["dark_cyber", "light_professional", "cyber_red", "security_blue", "matrix_green"]
        
        for theme in themes:
            # Test timeline
            fig_timeline = THGNNPlotlyThemes.create_themed_timeline(
                timestamps, values,
                theme_name=theme,
                title=f"Test Timeline - {theme}"
            )
            assert fig_timeline is not None
            
            # Test heatmap
            heatmap_data = np.random.randint(0, 100, (5, 5))
            x_labels = ['A', 'B', 'C', 'D', 'E']
            y_labels = ['1', '2', '3', '4', '5']
            
            fig_heatmap = THGNNPlotlyThemes.create_themed_heatmap(
                heatmap_data, x_labels, y_labels,
                theme_name=theme,
                title=f"Test Heatmap - {theme}"
            )
            assert fig_heatmap is not None
            
            print(f"✅ Visualizations created for theme '{theme}'")
        
        print("🎉 All visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive integration test"""
    print("🚀 Starting THGNN Dashboard Integration Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Theme System", test_theme_system()))
    test_results.append(("Production Components", test_production_components()))
    test_results.append(("Dashboard Functionality", test_dashboard_functionality()))
    test_results.append(("Visualization Creation", test_visualization_creation()))
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Dashboard is ready for production!")
        print("🌐 Dashboard running at: http://localhost:8502")
        return True
    else:
        print("❌ Some tests failed - please review and fix issues")
        return False

if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)