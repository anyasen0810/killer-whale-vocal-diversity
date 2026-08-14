# imports 
import pymupdf
import pandas as pd
import math
import numpy as np
from itertools import combinations
import os

# setting up directories 
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
output_mtdna = os.path.join(base_dir, "mtDNA.csv")

# opening pdf 
doc = pymupdf.open("Population_Structure.pdf")
# finding page with mtdna phylogram 
page = doc[33] 

drawings = page.get_drawings()
extracted_segments = []

for path_idx, draw in enumerate(drawings):
    # looking for lines 
    if "items" in draw:
        for item in draw["items"]:
            if item[0] == "l":  # Line segment
                p1, p2 = item[1], item[2]
                # saving beginning and end coordinates of line segments 
                extracted_segments.append({
                    "path_id": path_idx,
                    "x0": round(p1.x, 2),
                    "y0": round(p1.y, 2),
                    "x1": round(p2.x, 2),
                    "y1": round(p2.y, 2),
                    "type": "Horizontal" if round(p1.y, 2) == round(p2.y, 2) else "Vertical"
                })

# converting to dataframe 
df_segments = pd.DataFrame(extracted_segments)
print(f"--- Found {len(df_segments)} total line segments across the 12 paths ---")

# finding and sorting by length to make manual matching easier 
df_segments["length"] = df_segments.apply(
    lambda row: math.hypot(row["x1"] - row["x0"], row["y1"] - row["y0"]),
    axis=1)
print(df_segments.sort_values("length"))

# converting pixel length to evolutionary distances for final matrix 
df_segments["evo_length"] = df_segments["length"] / 64.5
print(df_segments)

