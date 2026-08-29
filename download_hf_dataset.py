import os
from datasets import load_dataset
from PIL import Image

def main():
    print("Downloading 42,000 ISL images from HuggingFace (Hemg/Indian_sign_language_dataset)...")
    # Load dataset
    ds = load_dataset("Hemg/Indian_sign_language_dataset", split="train")
    
    save_dir = "dataset/split_dataset/train"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get class names
    features = ds.features['label']
    class_names = features.names
    
    for cls in class_names:
        os.makedirs(os.path.join(save_dir, cls.upper()), exist_ok=True)
        
    print(f"Dataset has {len(ds)} images across {len(class_names)} classes.")
    print("Saving to local folders...")
    
    for i, item in enumerate(ds):
        img = item['image']
        label_id = item['label']
        cls_name = class_names[label_id].upper()
        
        # Save image
        img_path = os.path.join(save_dir, cls_name, f"hf_{i}.jpg")
        img.save(img_path)
        
        if (i+1) % 5000 == 0:
            print(f"Saved {i+1} / {len(ds)} images...")
            
    print("Download and extraction complete!")
    print("Note: This dataset contains 35 classes (1-9, A-Z). It is missing '0'.")

if __name__ == "__main__":
    main()
