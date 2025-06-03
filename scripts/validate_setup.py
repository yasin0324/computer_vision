#!/usr/bin/env python3
"""
Project setup validation script.

This script validates that the project structure is correctly set up
and all dependencies are properly installed.
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "torch", "torchvision", "numpy", "pandas", "PIL", "cv2",
        "matplotlib", "seaborn", "sklearn", "tqdm", "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                importlib.import_module("PIL")
            elif package == "cv2":
                importlib.import_module("cv2")
            elif package == "sklearn":
                importlib.import_module("sklearn")
            elif package == "yaml":
                importlib.import_module("yaml")
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed")
        return True

def check_directory_structure():
    """Check if directory structure is correct."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        "src", "src/config", "src/data", "src/models", "src/training",
        "src/evaluation", "src/utils", "scripts", "notebooks", "tests",
        "docs", "configs", "data", "data/raw", "data/processed", 
        "outputs", "outputs/models", "outputs/logs", "outputs/results"
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("Create with: mkdir -p " + " ".join(missing_dirs))
        return False
    else:
        print("✅ Directory structure complete")
        return True

def check_config_files():
    """Check if configuration files exist."""
    print("\n⚙️  Checking configuration files...")
    
    config_files = [
        "src/config/config.py",
        "src/config/paths.py", 
        "requirements.txt",
        ".gitignore"
    ]
    
    missing_files = []
    
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing configuration files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ Configuration files present")
        return True

def check_data_availability():
    """Check if data is available."""
    print("\n📊 Checking data availability...")
    
    plantvillage_path = Path("data/raw/PlantVillage")
    processed_path = Path("data/processed/splits")
    
    if plantvillage_path.exists():
        # Count subdirectories (classes)
        subdirs = [d for d in plantvillage_path.iterdir() if d.is_dir()]
        print(f"✅ PlantVillage dataset found ({len(subdirs)} classes)")
        
        # Check for target classes
        target_classes = [
            "Tomato_Bacterial_spot", "Tomato_Septoria_leaf_spot",
            "Tomato__Target_Spot", "Tomato_healthy"
        ]
        
        found_classes = []
        for class_name in target_classes:
            class_path = plantvillage_path / class_name
            if class_path.exists():
                image_count = len(list(class_path.glob("*.jpg"))) + len(list(class_path.glob("*.png")))
                print(f"  ✅ {class_name}: {image_count} images")
                found_classes.append(class_name)
            else:
                print(f"  ❌ {class_name}: Not found")
        
        if len(found_classes) == len(target_classes):
            print("✅ All target classes available")
            data_available = True
        else:
            print(f"⚠️  Missing {len(target_classes) - len(found_classes)} target classes")
            data_available = False
    else:
        print("❌ PlantVillage dataset not found")
        print("   Download from: https://www.kaggle.com/datasets/arjuntejaswi/plant-village")
        print("   Extract to: data/raw/PlantVillage/")
        data_available = False
    
    # Check processed data
    if processed_path.exists():
        split_files = ["train_split.csv", "val_split.csv", "test_split.csv"]
        found_splits = [f for f in split_files if (processed_path / f).exists()]
        
        if len(found_splits) == len(split_files):
            print("✅ Processed data splits found")
        else:
            print(f"⚠️  Missing data splits: {set(split_files) - set(found_splits)}")
            print("   Run: python scripts/preprocess_data.py")
    else:
        print("⚠️  No processed data found")
        print("   Run: python scripts/preprocess_data.py")
    
    return data_available

def check_imports():
    """Check if project modules can be imported."""
    print("\n🔗 Checking module imports...")
    
    modules_to_test = [
        "src.config.config",
        "src.config.paths", 
        "src.utils.logger"
    ]
    
    import_errors = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            import_errors.append(module)
    
    if import_errors:
        print(f"\n⚠️  Import errors in: {', '.join(import_errors)}")
        return False
    else:
        print("✅ All modules importable")
        return True

def run_quick_test():
    """Run a quick functionality test."""
    print("\n🧪 Running quick functionality test...")
    
    try:
        from src.config import config, paths
        from src.utils.logger import setup_logger
        
        # Test configuration
        print(f"✅ Config loaded - {config.NUM_CLASSES} classes")
        
        # Test paths
        print(f"✅ Paths loaded - Project root: {paths.PROJECT_ROOT}")
        
        # Test logger
        logger = setup_logger("test")
        logger.info("Test log message")
        print("✅ Logger working")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 TOMATO SPOT RECOGNITION - PROJECT SETUP VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_config_files),
        ("Data Availability", check_data_availability),
        ("Module Imports", check_imports),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Project is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run data preprocessing: python scripts/preprocess_data.py")
        print("2. Start training: python scripts/train_baseline.py")
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 