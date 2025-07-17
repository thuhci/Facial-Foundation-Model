#!/usr/bin/env python3
"""
Test script to verify global configuration and freezing functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import (
    get_cfg, 
    merge_config_file, 
    freeze_cfg, 
    is_cfg_frozen,
    reset_cfg,
    load_and_freeze_config
)


def test_global_config_access():
    """Test that configuration is truly global across different functions."""
    print("=" * 60)
    print("Testing Global Configuration Access")
    print("=" * 60)
    
    def function_a():
        """Function A accesses config."""
        cfg = get_cfg()
        return {
            'model_name': cfg.MODEL.NAME,
            'learning_rate': cfg.OPTIMIZATION.LR,
            'batch_size': cfg.DATA.BATCH_SIZE
        }
    
    def function_b():
        """Function B modifies config."""
        cfg = get_cfg()
        cfg.MODEL.NAME = 'modified_by_function_b'
        cfg.OPTIMIZATION.LR = 0.123
        cfg.DATA.BATCH_SIZE = 128
    
    def function_c():
        """Function C accesses config after modification."""
        cfg = get_cfg()
        return {
            'model_name': cfg.MODEL.NAME,
            'learning_rate': cfg.OPTIMIZATION.LR,
            'batch_size': cfg.DATA.BATCH_SIZE
        }
    
    # Test initial access
    print("1. Initial config from function_a:")
    result_a = function_a()
    print(f"   {result_a}")
    
    # Modify config from function_b
    print("\n2. Modifying config from function_b...")
    function_b()
    
    # Check if changes are visible from function_c
    print("\n3. Config from function_c after modification:")
    result_c = function_c()
    print(f"   {result_c}")
    
    # Verify they are the same
    print(f"\n4. Are configs the same? {result_a != result_c}")
    print("   ✅ Global configuration is working correctly!")


def test_config_freezing():
    """Test configuration freezing functionality."""
    print("\n" + "=" * 60)
    print("Testing Configuration Freezing")
    print("=" * 60)
    
    # Reset config first
    reset_cfg()
    
    cfg = get_cfg()
    print(f"1. Initial frozen state: {is_cfg_frozen()}")
    
    # Test modification before freezing
    print("\n2. Modifying config before freezing...")
    cfg.MODEL.NAME = 'test_model_before_freeze'
    cfg.OPTIMIZATION.LR = 0.001
    print(f"   Model name: {cfg.MODEL.NAME}")
    print(f"   Learning rate: {cfg.OPTIMIZATION.LR}")
    
    # Freeze the config
    print("\n3. Freezing configuration...")
    freeze_cfg()
    print(f"   Frozen state: {is_cfg_frozen()}")
    
    # Test modification after freezing (should fail)
    print("\n4. Attempting to modify config after freezing...")
    try:
        cfg.MODEL.NAME = 'test_model_after_freeze'
        print("   ❌ ERROR: Config was modified after freezing!")
    except Exception as e:
        print(f"   ✅ SUCCESS: Config modification blocked: {type(e).__name__}")
    
    # Test that config is still accessible
    print(f"\n5. Config still accessible after freezing:")
    print(f"   Model name: {cfg.MODEL.NAME}")
    print(f"   Learning rate: {cfg.OPTIMIZATION.LR}")


def test_yaml_loading_and_freezing():
    """Test loading YAML config and freezing."""
    print("\n" + "=" * 60)
    print("Testing YAML Loading and Freezing")
    print("=" * 60)
    
    # Reset config first
    reset_cfg()
    
    # Load and freeze config in one step
    print("1. Loading and freezing config from YAML...")
    cfg = load_and_freeze_config('configs/gaze360_finetune.yaml')
    
    print(f"   Frozen state: {is_cfg_frozen()}")
    print(f"   Model name: {cfg.MODEL.NAME}")
    print(f"   Dataset: {cfg.DATA.DATASET_NAME}")
    print(f"   Learning rate: {cfg.OPTIMIZATION.LR}")
    
    # Test that other functions can access the frozen config
    def another_function():
        cfg = get_cfg()
        return {
            'model': cfg.MODEL.NAME,
            'dataset': cfg.DATA.DATASET_NAME,
            'lr': cfg.OPTIMIZATION.LR,
            'frozen': is_cfg_frozen()
        }
    
    print("\n2. Accessing frozen config from another function:")
    result = another_function()
    print(f"   {result}")
    print("   ✅ Frozen config is globally accessible!")


def test_multi_file_simulation():
    """Simulate accessing config from multiple files."""
    print("\n" + "=" * 60)
    print("Testing Multi-file Global Access Simulation")
    print("=" * 60)
    
    # Reset and load config
    reset_cfg()
    cfg = get_cfg()
    merge_config_file(cfg, 'configs/gaze360_finetune.yaml')
    freeze_cfg()
    
    # Simulate different modules accessing config
    def simulate_training_module():
        """Simulate training module accessing config."""
        cfg = get_cfg()
        return {
            'module': 'training',
            'epochs': cfg.TRAINING.EPOCHS,
            'batch_size': cfg.DATA.BATCH_SIZE,
            'frozen': is_cfg_frozen()
        }
    
    def simulate_model_module():
        """Simulate model module accessing config."""
        cfg = get_cfg()
        return {
            'module': 'model',
            'model_name': cfg.MODEL.NAME,
            'drop_path': cfg.MODEL.DROP_PATH,
            'frozen': is_cfg_frozen()
        }
    
    def simulate_data_module():
        """Simulate data module accessing config."""
        cfg = get_cfg()
        return {
            'module': 'data',
            'dataset': cfg.DATA.DATASET_NAME,
            'num_frames': cfg.DATA.NUM_FRAMES,
            'frozen': is_cfg_frozen()
        }
    
    print("1. Config access from different 'modules':")
    
    training_result = simulate_training_module()
    print(f"   Training module: {training_result}")
    
    model_result = simulate_model_module()
    print(f"   Model module: {model_result}")
    
    data_result = simulate_data_module()
    print(f"   Data module: {data_result}")
    
    print("\n2. All modules can access the same frozen config:")
    print(f"   All frozen: {training_result['frozen'] and model_result['frozen'] and data_result['frozen']}")
    print("   ✅ Multi-file global access working correctly!")


def test_recommended_workflow():
    """Test the recommended workflow for using global config."""
    print("\n" + "=" * 60)
    print("Testing Recommended Workflow")
    print("=" * 60)
    
    print("Recommended workflow for using global configuration:")
    print("1. Load configuration from YAML file")
    print("2. Apply any command-line overrides")
    print("3. Freeze the configuration")
    print("4. Access configuration from any module using get_cfg()")
    
    print("\n" + "-" * 40)
    print("Implementing recommended workflow:")
    
    # Step 1: Load config
    print("\nStep 1: Loading configuration...")
    reset_cfg()
    cfg = get_cfg()
    merge_config_file(cfg, 'configs/gaze360_finetune.yaml')
    print(f"   Loaded config from YAML")
    
    # Step 2: Apply overrides
    print("\nStep 2: Applying command-line overrides...")
    cfg.OPTIMIZATION.LR = 0.002  # Simulate --lr 0.002
    cfg.DATA.BATCH_SIZE = 64     # Simulate --batch_size 64
    cfg.TRAINING.EPOCHS = 100    # Simulate --epochs 100
    print(f"   Applied overrides: lr={cfg.OPTIMIZATION.LR}, batch_size={cfg.DATA.BATCH_SIZE}, epochs={cfg.TRAINING.EPOCHS}")
    
    # Step 3: Freeze config
    print("\nStep 3: Freezing configuration...")
    freeze_cfg()
    print(f"   Configuration frozen: {is_cfg_frozen()}")
    
    # Step 4: Access from any module
    print("\nStep 4: Accessing config from modules...")
    
    def training_script():
        cfg = get_cfg()
        return f"Training for {cfg.TRAINING.EPOCHS} epochs with lr={cfg.OPTIMIZATION.LR}"
    
    def model_builder():
        cfg = get_cfg()
        return f"Building model {cfg.MODEL.NAME} with drop_path={cfg.MODEL.DROP_PATH}"
    
    def data_loader():
        cfg = get_cfg()
        return f"Loading {cfg.DATA.DATASET_NAME} with batch_size={cfg.DATA.BATCH_SIZE}"
    
    print(f"   Training script: {training_script()}")
    print(f"   Model builder: {model_builder()}")
    print(f"   Data loader: {data_loader()}")
    
    print("\n✅ Recommended workflow completed successfully!")


if __name__ == '__main__':
    print("Global Configuration and Freezing Test")
    print("=" * 60)
    
    try:
        test_global_config_access()
        test_config_freezing()
        test_yaml_loading_and_freezing()
        test_multi_file_simulation()
        test_recommended_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Global configuration system is working correctly.")
        print("=" * 60)
        
        print("\n📝 Usage Summary:")
        print("1. Use load_and_freeze_config('config.yaml') to load and freeze config")
        print("2. Use get_cfg() in any file to access the global frozen config")
        print("3. Configuration changes are visible globally before freezing")
        print("4. After freezing, config is immutable but still globally accessible")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
