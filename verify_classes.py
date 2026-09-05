#!/usr/bin/env python3
from roboflow import Roboflow
import sys

rf = Roboflow(api_key='MyrR6UFnOMOjR8gHw6MI')
project = rf.workspace().project('waterdrain')
classes = project.classes
class_set = set(classes)

expected = {"drain", "blockage_area"}
print(f"Waterdrain project classes: {classes}")
print(f"Class set: {class_set}")
print(f"Expected: {expected}")
print(f"Match: {class_set == expected}")

if class_set == expected:
    print("\n✓ Project schema is correct! Ready for data upload.")
    sys.exit(0)
else:
    print("\n✗ Project schema mismatch. Please reset classes first.")
    sys.exit(1)