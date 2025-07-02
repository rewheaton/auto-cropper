#!/usr/bin/env python3
"""
Test script to verify error handling functionality.
"""

import tempfile
import os
from pathlib import Path
from auto_cropper.cli import validate_input_file, ensure_output_directory, validate_json_file
import click

def test_error_handling():
    """Test various error scenarios."""
    print("🧪 Testing error handling scenarios...")
    
    # Test 1: Non-existent file
    print("\n1. Testing non-existent file...")
    try:
        validate_input_file("nonexistent_file.mp4")
        print("❌ Should have failed!")
    except click.ClickException as e:
        print(f"✅ Correctly caught: {e}")
    
    # Test 2: Unsupported file format
    print("\n2. Testing unsupported file format...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test content")
        temp_txt = f.name
    
    try:
        validate_input_file(temp_txt)
        print("❌ Should have failed!")
    except click.ClickException as e:
        print(f"✅ Correctly caught: {e}")
    finally:
        os.unlink(temp_txt)
    
    # Test 3: Directory instead of file
    print("\n3. Testing directory instead of file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            validate_input_file(temp_dir)
            print("❌ Should have failed!")
        except click.ClickException as e:
            print(f"✅ Correctly caught: {e}")
    
    # Test 4: Permission denied directory
    print("\n4. Testing permission denied (simulated)...")
    try:
        ensure_output_directory("/root/no_permission_dir")
        print("❌ Should have failed!")
    except click.ClickException as e:
        print(f"✅ Correctly caught: {e}")
    
    # Test 5: Invalid JSON file
    print("\n5. Testing invalid JSON file...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"invalid json content {")
        temp_json = f.name
    
    try:
        validate_json_file(temp_json)
        print("❌ Should have failed!")
    except click.ClickException as e:
        print(f"✅ Correctly caught: {e}")
    finally:
        os.unlink(temp_json)
    
    # Test 6: JSON missing required keys
    print("\n6. Testing JSON missing required keys...")
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
        json.dump({"wrong_key": "value"}, f)
        temp_json = f.name
    
    try:
        validate_json_file(temp_json, ['video_info', 'frames'])
        print("❌ Should have failed!")
    except click.ClickException as e:
        print(f"✅ Correctly caught: {e}")
    finally:
        os.unlink(temp_json)
    
    # Test 7: Successful validation
    print("\n7. Testing successful validation...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake video content")
        temp_mp4 = f.name
    
    try:
        result = validate_input_file(temp_mp4)
        print(f"✅ Successfully validated: {result}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        os.unlink(temp_mp4)
    
    print("\n🎉 Error handling tests completed!")

if __name__ == "__main__":
    test_error_handling()
