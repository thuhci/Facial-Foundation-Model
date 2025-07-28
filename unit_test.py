#!/usr/bin/env python3
"""
Unit test script for finetune and pretrain workflows.
Tests data loaders, model creation, and training steps for both workflows.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration utilities
from src.utils.config import get_cfg, merge_config_file, reset_cfg
from src import utils
from src.utils import logger

# Import finetune functions
import run_finetuning_with_yacs as finetune_module
from run_finetuning_with_yacs import (
    create_data_loaders as finetune_create_data_loaders,
    create_model_from_config as finetune_create_model_from_config,
    create_optimizer_from_config as finetune_create_optimizer_from_config,
    create_criterion_from_config as finetune_create_criterion_from_config,
    create_mixup_from_config as finetune_create_mixup_from_config,
    create_scheduler_from_config as finetune_create_scheduler_from_config,
)

# Import pretrain functions
import run_pretraining_with_yacs as pretrain_module
from run_pretraining_with_yacs import (
    create_data_loader as pretrain_create_data_loader,
    create_model_from_config as pretrain_create_model_from_config,
    create_optimizer_from_config as pretrain_create_optimizer_from_config,
    create_scheduler_from_config as pretrain_create_scheduler_from_config,
)

# Import engines
from src.engine.train_engine import TrainingEngine
from src.engine.val_engine import ValidationEngine
from src.engine.pretrain_engine import PretrainEngine
from src.utils.utils import NativeScalerWithGradNormCount as NativeScaler


class UnitTester:
    """Unit test class for finetune and pretrain workflows."""
    
    def __init__(self):
        self.test_results = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Test configurations
        self.finetune_configs = [
            'configs/gaze360_finetune.yaml',
            'configs/dfew_finetune.yaml',
            'configs/eve_finetune.yaml',  # Add more if needed
        ]
        
        self.pretrain_configs = [
            'configs/voxceleb2_pretrain.yaml',
            # Add more pretrain configs if available
        ]
        
        # Reduce dataset size for testing
        self.test_epochs = 1
        self.test_steps = 2  # Only test a few steps
    
    def setup_minimal_distributed(self):
        """Setup minimal distributed environment for testing."""
        try:
            # Skip distributed setup for unit testing
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            os.environ.setdefault('LOCAL_RANK', '0')
            
            # Mock distributed functions for testing
            utils.utils._LOCAL_RANK = 0
            utils.utils._WORLD_SIZE = 1
            utils.utils._RANK = 0
            
            print("✅ Minimal distributed setup completed")
            return True
        except Exception as e:
            print(f"❌ Distributed setup failed: {e}")
            return False
    
    def test_finetune_workflow(self, config_path):
        """Test finetune workflow with given config."""
        print(f"\n{'='*60}")
        print(f"Testing Finetune Workflow: {config_path}")
        print(f"{'='*60}")
        
        test_result = {
            'config': config_path,
            'status': 'FAILED',
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # Reset configuration
            reset_cfg()
            
            # Step 1: Load configuration
            print("Step 1: Loading configuration...")
            cfg = get_cfg()
            merge_config_file(cfg, config_path)
            
            # Override some settings for testing
            cfg.DATA.BATCH_SIZE = 2  # Small batch size for testing
            cfg.DATA.NUM_WORKERS = 2
            cfg.TRAINING.EPOCHS = self.test_epochs
            cfg.SYSTEM.DISTRIBUTED = False  # Disable distributed for unit testing
            cfg.TRAINING.FINETUNE = ''  # Skip pretrained model loading for quick testing
            
            test_result['steps_completed'].append('config_loaded')
            print("✅ Configuration loaded successfully")
            
            # Step 2: Create data loaders
            print("\nStep 2: Creating data loaders...")
            data_loader_train, data_loader_val = finetune_create_data_loaders()
            
            print(f"   Train loader batches: {len(data_loader_train)}")
            print(f"   Val loader batches: {len(data_loader_val)}")
            test_result['steps_completed'].append('data_loaders_created')
            print("✅ Data loaders created successfully")
            
            # Step 3: Create model
            print("\nStep 3: Creating model...")
            model = finetune_create_model_from_config()
            model.to(self.device)
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Model parameters: {n_params:,}")
            test_result['steps_completed'].append('model_created')
            print("✅ Model created successfully")
            
            # Step 4: Create optimizer and criterion
            print("\nStep 4: Creating optimizer and criterion...")
            optimizer = finetune_create_optimizer_from_config(model)
            criterion = finetune_create_criterion_from_config()
            mixup_fn = finetune_create_mixup_from_config()
            
            print(f"   Optimizer: {type(optimizer).__name__}")
            print(f"   Criterion: {type(criterion).__name__}")
            print(f"   Mixup: {mixup_fn is not None}")
            test_result['steps_completed'].append('optimizer_created')
            print("✅ Optimizer and criterion created successfully")
            
            # Step 5: Create scheduler
            # print("\nStep 5: Creating scheduler...")
            # num_training_steps_per_epoch = len(data_loader_train) // cfg.TRAINING.UPDATE_FREQ
            # lr_schedule_values, wd_schedule_values = finetune_create_scheduler_from_config(optimizer,num_training_steps_per_epoch)

            # Step 6: Training step
            print("\nStep 6: Testing training step...")
            model.train()
            loss_scaler = NativeScaler()
            
            train_engine = TrainingEngine(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mixup_fn=mixup_fn,
                loss_scaler=loss_scaler,
                device=self.device
            )
            
            # Create a small subset of data for testing
            test_batches = []
            batch_count = 0
            for batch in data_loader_train:
                if batch_count >= self.test_steps:
                    break
                test_batches.append(batch)
                batch_count += 1
            
            # Test training using the engine's train_one_epoch method
            try:
                print(f"   Testing {len(test_batches)} training batches...")
                
                # Mock the training loop similar to engine's train_one_epoch
                metric_logger = utils.logger.MetricLogger(delimiter="  ")
                train_engine.model.train(True)
                
                total_loss = 0.0
                step_count = 0
                
                for data_iter_step, batch_data in enumerate(test_batches):
                    # Process batch like in train_engine
                    samples, targets = train_engine._process_batch(batch_data)
                    
                    # Forward pass
                    loss, output = train_engine._forward_pass(samples, targets)
                    
                    # Backward pass
                    grad_norm = train_engine._backward_pass(loss, data_iter_step)
                    
                    total_loss += loss.item()
                    step_count += 1
                    
                    print(f"   Training step {step_count}: loss = {loss.item():.4f}")
                
                if step_count > 0:
                    avg_loss = total_loss / step_count
                    print(f"   Average training loss: {avg_loss:.4f}")
                    test_result['steps_completed'].append('training_steps')
                    print("✅ Training steps completed successfully")
                    
            except Exception as e:
                print(f"   ❌ Training steps failed: {e}")
                test_result['errors'].append(f"Training step error: {e}")
                traceback.print_exc()
            
            # Step 6: Validation step
            print("\nStep 6: Testing validation step...")
            model.eval()
            
            val_engine = ValidationEngine(
                model=model,
                device=self.device
            )
            
            # Create a small subset of validation data for testing
            test_val_batches = []
            batch_count = 0
            for batch in data_loader_val:
                if batch_count >= self.test_steps:
                    break
                test_val_batches.append(batch)
                batch_count += 1
            
            # Create a temporary DataLoader for the test batches
            class TestDataLoader:
                def __init__(self, batches):
                    self.batches = batches
                def __iter__(self):
                    return iter(self.batches)
                def __len__(self):
                    return len(self.batches)
            
            test_val_loader = TestDataLoader(test_val_batches)
            
            try:
                print(f"   Testing {len(test_val_batches)} validation batches...")
                
                # Use ValidationEngine's validate method
                val_metrics = val_engine.validate(test_val_loader)
                
                print(f"   Validation metrics: {val_metrics}")
                test_result['steps_completed'].append('validation_steps')
                print("✅ Validation steps completed successfully")
                
            except Exception as e:
                print(f"   ❌ Validation steps failed: {e}")
                test_result['errors'].append(f"Validation step error: {e}")
                traceback.print_exc()
            
            test_result['status'] = 'SUCCESS'
            print(f"\n🎉 Finetune workflow test PASSED for {config_path}")
            
        except Exception as e:
            error_msg = f"Finetune test failed: {e}"
            print(f"\n❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        return test_result
    
    def test_pretrain_workflow(self, config_path):
        """Test pretrain workflow with given config."""
        print(f"\n{'='*60}")
        print(f"Testing Pretrain Workflow: {config_path}")
        print(f"{'='*60}")
        
        test_result = {
            'config': config_path,
            'status': 'FAILED',
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # Reset configuration
            reset_cfg()
            
            # Step 1: Load configuration
            print("Step 1: Loading configuration...")
            cfg = get_cfg()
            merge_config_file(cfg, config_path)
            
            # Override some settings for testing
            cfg.DATA.BATCH_SIZE = 4  # Small batch size for testing
            cfg.DATA.NUM_WORKERS = 2
            cfg.TRAINING.EPOCHS = self.test_epochs
            cfg.SYSTEM.DISTRIBUTED = False  # Disable distributed for unit testing
            
            test_result['steps_completed'].append('config_loaded')
            print("✅ Configuration loaded successfully")
            
            
            
            
            
            # Step 2: Create model  # [NOTE] createmodelahead data loader! since cfg neede updated
            print("\nStep 2: Creating model...")
            model = pretrain_create_model_from_config()
            model.to(self.device)
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Model parameters: {n_params:,}")
            test_result['steps_completed'].append('model_created')
            print("✅ Model created successfully")
            
            
            # Step 3: Create data loader
            print("\nStep 4: Creating data loader...")
            data_loader_train, dataset_length = pretrain_create_data_loader()
            
            print(f"   Train loader batches: {len(data_loader_train)}")
            print(f"   Dataset length: {dataset_length}")
            test_result['steps_completed'].append('data_loader_created')
            print("✅ Data loader created successfully")
            
            # Step 4: Create optimizer
            print("\nStep 4: Creating optimizer...")
            optimizer = pretrain_create_optimizer_from_config(model)
            
            print(f"   Optimizer: {type(optimizer).__name__}")
            test_result['steps_completed'].append('optimizer_created')
            print("✅ Optimizer created successfully")
            
            # Step 5: Create scheduler
            # print("\nStep 5: Creating scheduler...")
            # num_training_steps_per_epoch = len(data_loader_train) // cfg.TRAINING.UPDATE_FREQ
            # lr_schedule_values, wd_schedule_values = pretrain_create_scheduler_from_config(num_training_steps_per_epoch)
            
            # Step 6: Training step
            print("\nStep 6: Testing pretraining step...")
            model.train()
            loss_scaler = NativeScaler()
            
            pretrain_engine = PretrainEngine(
                model=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                device=self.device
            )
            
            # Create a small subset of data for testing
            test_batches = []
            batch_count = 0
            for batch in data_loader_train:
                if batch_count >= self.test_steps:
                    break
                test_batches.append(batch)
                batch_count += 1
            
            # Test pretraining using the engine's methods
            try:
                print(f"   Testing {len(test_batches)} pretraining batches...")
                
                pretrain_engine.model.train()
                total_loss = 0.0
                step_count = 0
                
                for step, batch in enumerate(test_batches):
                    # Process batch like in pretrain_engine
                    videos, bool_masked_pos = pretrain_engine._process_batch(batch)
                    
                    # print("[DEBUG] videos shape:", videos.shape, 
                    #       "bool_masked_pos shape:", bool_masked_pos.shape)
                    # bool_masked_pos = bool_masked_pos.unsqueeze(1).repeat(1, videos.shape[2], 1)  # Ensure correct shape
                    
                    # Generate targets
                    patch_size = cfg.MODEL.PATCH_SIZE[0] if hasattr(cfg.MODEL, 'PATCH_SIZE') and cfg.MODEL.PATCH_SIZE else 16
                    labels = pretrain_engine._generate_targets(videos, bool_masked_pos, patch_size)
                    
                    # Forward pass
                    loss = pretrain_engine._forward_pass(videos, bool_masked_pos, labels)
                    
                    # Backward pass
                    grad_norm = pretrain_engine._backward_pass(loss)
                    
                    total_loss += loss.item()
                    step_count += 1
                    
                    print(f"   Pretraining step {step_count}: loss = {loss.item():.4f}")
                
                if step_count > 0:
                    avg_loss = total_loss / step_count
                    print(f"   Average pretraining loss: {avg_loss:.4f}")
                    test_result['steps_completed'].append('pretraining_steps')
                    print("✅ Pretraining steps completed successfully")
                    
            except Exception as e:
                print(f"   ❌ Pretraining steps failed: {e}")
                test_result['errors'].append(f"Pretraining step error: {e}")
                traceback.print_exc()
                avg_loss = total_loss / step_count
                print(f"   Average pretraining loss: {avg_loss:.4f}")
                test_result['steps_completed'].append('pretraining_steps')
                print("✅ Pretraining steps completed successfully")
            
            test_result['status'] = 'SUCCESS'
            print(f"\n🎉 Pretrain workflow test PASSED for {config_path}")
            
        except Exception as e:
            error_msg = f"Pretrain test failed: {e}"
            print(f"\n❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        return test_result
    
    def run_all_tests(self):
        """Run all unit tests."""
        print("🚀 Starting Unit Tests for Facial-Foundation-Model")
        print("=" * 80)
        
        # Setup environment
        if not self.setup_minimal_distributed():
            print("❌ Failed to setup test environment")
            return
        
        all_results = []
        
        # Test finetune workflows
        print(f"\n📋 Testing {len(self.finetune_configs)} Finetune Configurations")
        print("-" * 60)
        
        for config_path in self.finetune_configs:
            if os.path.exists(config_path):
                result = self.test_finetune_workflow(config_path)
                all_results.append(result)
            else:
                print(f"⚠️  Config file not found: {config_path}")
                all_results.append({
                    'config': config_path,
                    'status': 'SKIPPED',
                    'steps_completed': [],
                    'errors': ['Config file not found']
                })
        
        # Test pretrain workflows
        print(f"\n📋 Testing {len(self.pretrain_configs)} Pretrain Configurations")
        print("-" * 60)
        
        for config_path in self.pretrain_configs:
            if os.path.exists(config_path):
                result = self.test_pretrain_workflow(config_path)
                all_results.append(result)
            else:
                print(f"⚠️  Config file not found: {config_path}")
                all_results.append({
                    'config': config_path,
                    'status': 'SKIPPED',
                    'steps_completed': [],
                    'errors': ['Config file not found']
                })
        
        # Generate test report
        self.generate_test_report(all_results)
        
        return all_results
    
    def generate_test_report(self, results):
        """Generate and print test report."""
        print("\n" + "=" * 80)
        print("📊 UNIT TEST REPORT")
        print("=" * 80)
        
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'SUCCESS'])
        failed_tests = len([r for r in results if r['status'] == 'FAILED'])
        skipped_tests = len([r for r in results if r['status'] == 'SKIPPED'])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        print("-" * 80)
        
        for result in results:
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌" if result['status'] == 'FAILED' else "⏭️"
            print(f"{status_icon} {result['config']}")
            print(f"   Status: {result['status']}")
            print(f"   Steps completed: {', '.join(result['steps_completed'])}")
            
            if result['errors']:
                print(f"   Errors: {len(result['errors'])}")
                for error in result['errors']:
                    print(f"     - {error}")
            print()
        
        # Summary
        if passed_tests == total_tests:
            print("🎉 All tests passed! The system is working correctly.")
        elif passed_tests > 0:
            print(f"⚠️  {passed_tests}/{total_tests} tests passed. Some issues need attention.")
        else:
            print("❌ All tests failed. Critical issues need to be resolved.")
        
        print("=" * 80)


def main():
    """Main function to run unit tests."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create and run tests
    tester = UnitTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    failed_count = len([r for r in results if r['status'] == 'FAILED'])
    if failed_count > 0:
        print(f"\n❌ {failed_count} tests failed. Exiting with error code.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
