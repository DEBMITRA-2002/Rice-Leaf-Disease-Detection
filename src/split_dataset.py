from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("data/raw/Rice_Leaf_AUG")
OUTPUT_DIR = Path("data/processed")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

random.seed(SEED)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR.resolve()}")

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"{OUTPUT_DIR} is not empty. "
        "Move or rename it before creating a new split."
    )

class_folders = sorted(
    folder for folder in SOURCE_DIR.iterdir() if folder.is_dir()
)

if not class_folders:
    raise ValueError("No class folders found in the dataset.")

for class_folder in class_folders:
    images = sorted(
        path for path in class_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    random.shuffle(images)

    n = len(images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    print(f"\n{class_folder.name}: {n} images")

    for split_name, split_images in splits.items():
        destination = OUTPUT_DIR / split_name / class_folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, destination / image_path.name)

        print(f"  {split_name}: {len(split_images)}")

print("\nDataset split completed.")
print(f"Output: {OUTPUT_DIR.resolve()}")
