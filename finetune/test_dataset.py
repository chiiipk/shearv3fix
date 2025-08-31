#!/usr/bin/env python3
"""
Test script to verify STS-B dataset loading from HuggingFace
"""

from data_loader import get_sts_info, load_sts_dataset

def test_dataset_loading():
    """Test that STS-B dataset loads correctly"""
    print("🧪 Testing STS-B dataset loading...")
    
    try:
        # Test dataset info
        info = get_sts_info()
        print(f"✅ Dataset info loaded successfully")
        print(f"   Dataset: {info['dataset_name']}")
        print(f"   Train size: {info['train_size']:,}")
        print(f"   Validation size: {info['validation_size']:,}")
        print(f"   Test size: {info['test_size']:,}")
        
        # Show example
        example = info['example']
        print(f"\n📝 Example:")
        print(f"   Sentence 1: {example['sentence1']}")
        print(f"   Sentence 2: {example['sentence2']}")
        print(f"   Similarity: {example['label']:.2f}")
        
        # Test full dataset loading
        dataset = load_sts_dataset()
        print(f"\n✅ Full dataset loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    exit(0 if success else 1)
