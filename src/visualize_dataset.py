from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = Path("data/raw/Rice_Leaf_AUG")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR.resolve()}")

classes = {}
for folder in sorted(DATASET_DIR.iterdir()):
    if folder.is_dir():
        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        if images:
            classes[folder.name] = images

if not classes:
    raise ValueError("No images found. Check the dataset folder structure.")

print("Classes and image counts:")
for name, images in classes.items():
    print(f"{name}: {len(images)}")

# Class distribution chart
plt.figure(figsize=(10, 5))
plt.bar(list(classes.keys()), [len(v) for v in classes.values()])
plt.title("Rice Leaf Disease - Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.show()

# Sample images: one image per class
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, images) in zip(axes, classes.items()):
    image_path = random.choice(images)
    try:
        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(name)
        ax.axis("off")
    except Exception as error:
        ax.set_title(f"{name}\nCould not load")
        ax.axis("off")
        print(f"Error loading {image_path}: {error}")

for ax in axes[len(classes):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_images.png", dpi=150)
plt.show()

print("\nSaved charts in the outputs/ folder.")
