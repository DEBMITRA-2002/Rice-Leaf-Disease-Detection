from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_DIR = Path("data/raw/Rice_Leaf_AUG")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR.resolve()}"
    )

class_counts = Counter()
image_sizes = Counter()
corrupted_images = []
total_images = 0

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    count = 0

    for image_path in class_dir.rglob("*"):
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                image_sizes[img.size] += 1

            count += 1
            total_images += 1

        except Exception as error:
            corrupted_images.append(str(image_path))
            print(f"Unreadable: {image_path} | {error}")

    class_counts[class_dir.name] = count

print("\n===== DATASET SUMMARY =====")
print("Total images:", total_images)
print("Number of classes:", len(class_counts))

print("\nImages per class:")
for name, count in class_counts.items():
    print(f"{name}: {count}")

print("\nMost common image sizes:")
for size, count in image_sizes.most_common(10):
    print(f"{size}: {count}")

print("\nUnreadable images:", len(corrupted_images))
