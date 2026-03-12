import os
import shutil
import random

def split_dataset(source_folder, train_folder, val_folder, split=0.8):
    images = os.listdir(source_folder)
    random.shuffle(images)
    
    split_index = int(len(images) * split)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # Copy to train
    for img in train_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(train_folder, img)
        )
    
    # Copy to val
    for img in val_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(val_folder, img)
        )
    
    print(f"Train: {len(train_images)} images")
    print(f"Val:   {len(val_images)} images")



# Non Drowsy → alert class
print("Processing Non Drowsy (alert) images...")
split_dataset(
    source_folder=r"D:\Drowsiness-detection\data\raw\Driver Drowsiness Dataset (DDD)\Non Drowsy",
    train_folder=r"D:\Drowsiness-detection\data\images\train\alert",
    val_folder=r"D:\Drowsiness-detection\data\images\val\alert"
)

# Drowsy → drowsy class
print("Processing Drowsy images...")
split_dataset(
    source_folder=r"D:\Drowsiness-detection\data\raw\Driver Drowsiness Dataset (DDD)\Drowsy",
    train_folder=r"D:\Drowsiness-detection\data\images\train\drowsy",
    val_folder=r"D:\Drowsiness-detection\data\images\val\drowsy"
)

print("\nDataset preparation complete!")