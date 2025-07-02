#!/usr/bin/env python3
"""
Verification script to check that all pinned dependencies are correctly installed.
Run this after installing requirements.txt to verify everything works.
"""

import sys
from importlib.metadata import version

def check_version(package_name, expected_version):
    """Check if installed package version matches expected version."""
    try:
        installed_version = version(package_name)
        if installed_version == expected_version:
            print(f"✅ {package_name}: {installed_version}")
            return True
        else:
            print(f"❌ {package_name}: expected {expected_version}, got {installed_version}")
            return False
    except Exception as e:
        print(f"❌ {package_name}: not installed or error ({e})")
        return False

def main():
    """Main verification function."""
    print("🔍 Verifying pinned dependencies...")
    
    # Core dependencies from requirements.txt
    dependencies = {
        "click": "8.2.1",
        "opencv-python": "4.11.0.86", 
        "numpy": "2.3.1",
        "ultralytics": "8.3.161",
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "tqdm": "4.67.1"
    }
    
    all_good = True
    for package, expected_version in dependencies.items():
        if not check_version(package, expected_version):
            all_good = False
    
    if all_good:
        print("\n🎉 All dependencies verified successfully!")
        return 0
    else:
        print("\n💥 Some dependencies don't match expected versions.")
        print("Try: pip install -r requirements.txt --force-reinstall")
        return 1

if __name__ == "__main__":
    sys.exit(main())
