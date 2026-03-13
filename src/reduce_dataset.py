import os
import shutil
import random

def copy_subset(source, destination, count):
    os.makedirs(destination, exist_ok=True)
    images = os.listdir(source)
    random.shuffle(images)
    selected = images[:count]
    for img in selected:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(destination, img)
        )
    print(f"Copied {len(selected)} images → {destination}")

# Train - 3000 each
print("Processing train/alert...")
copy_subset(
    r"D:\Drowsiness-detection\data\images\train\alert",
    r"D:\Drowsiness-detection\data\small\train\alert",
    3000
)

print("Processing train/drowsy...")
copy_subset(
    r"D:\Drowsiness-detection\data\images\train\drowsy",
    r"D:\Drowsiness-detection\data\small\train\drowsy",
    3000
)

# Val - 500 each
print("Processing val/alert...")
copy_subset(
    r"D:\Drowsiness-detection\data\images\val\alert",
    r"D:\Drowsiness-detection\data\small\val\alert",
    500
)

print("Processing val/drowsy...")
copy_subset(
    r"D:\Drowsiness-detection\data\images\val\drowsy",
    r"D:\Drowsiness-detection\data\small\val\drowsy",
    500
)

print("\nSmall dataset ready! ")
print("Total: 7,000 images")