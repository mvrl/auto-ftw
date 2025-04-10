import pandas as pd

df = pd.read_csv("/home/p.vinh/Auto-FTW/combined.csv")

print(df["s2_scene_id"].nunique())