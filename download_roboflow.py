from roboflow import Roboflow
import os
import json
import yaml

os.environ['ROBOFLOW_API'] = 'MyrR6UFnOMOjR8gHw6MI'
rf = Roboflow(os.environ['ROBOFLOW_API'])
project = rf.workspace().project('waterdrain')

# List versions
print("Project versions:")
for version in project.versions():
    print(f"  Version {version.id}")
    
# Get version 1
latest = project.version(1)
print(f"\nVersion 1 splits: {latest.splits}")

# Download the dataset in coco-segmentation format
print("\nDownloading dataset in coco-segmentation format...")
dataset = latest.download("coco-segmentation")

# The dataset is extracted to a directory
export_dir = os.path.join(os.getcwd(), "waterDrain-1")
print(f"\nExport directory: {export_dir}")

# Check structure
if os.path.exists(export_dir):
    for root, dirs, files in os.walk(export_dir):
        level = root.replace(export_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            if file.endswith(".yaml") or file.endswith(".yml"):
                with open(filepath) as f:
                    data = yaml.safe_load(f)
                    print(f"{subindent}{file}: classes={data.get('nc', 'N/A')}, names={data.get('names', 'N/A')}")
            elif file.endswith(".json"):
                print(f"{subindent}{file}: found (COCO annotations)")
            else:
                print(f"{subindent}{file}")