import os

# === Project root directory ===
project_name = "modular_image_classification"

# === Folder structure ===
folders = [
    project_name,
    os.path.join(project_name, "utils"),
    os.path.join(project_name, "configs"),
    os.path.join(project_name, "part_1_results"),
    os.path.join(project_name, "part_2_results"),
]

# === Files to create ===
files = [
    os.path.join(project_name, "datasets.py"),
    os.path.join(project_name, "models.py"),
    os.path.join(project_name, "train.py"),
    os.path.join(project_name, "eval.py"),
    os.path.join(project_name, "export_onnx.py"),
    os.path.join(project_name, "quantize.py"),
    os.path.join(project_name, "requirements.txt"),
    os.path.join(project_name, "README.md"),
    # Utils
    os.path.join(project_name, "utils", "metrics.py"),
    os.path.join(project_name, "utils", "seed.py"),
    os.path.join(project_name, "utils", "preprocessing.py"),
    # Configs
    os.path.join(project_name, "configs", "train.yaml"),
    os.path.join(project_name, "configs", "model.yaml"),
    # Result placeholders
    os.path.join(project_name, "part_1_results", "summary.txt"),
    os.path.join(project_name, "part_2_results", "summary.txt"),
]

# === Create folders ===
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# === Create empty files ===
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("")  # empty file
        print(f"Created file: {file}")
    else:
        print(f"File already exists: {file}")

print("\n✅ Project skeleton created successfully!")
