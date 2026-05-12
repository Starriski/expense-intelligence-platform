print("SCRIPT STARTED")

import os
import pandas as pd

file_path = "data/raw/credit_card.csv"

print("Checking file exists...")

print("CWD:", os.getcwd())
print("File exists:", os.path.exists(file_path))

df = pd.read_csv(file_path)

print("DATA LOADED SUCCESSFULLY")
print("Shape:", df.shape)

print(df.head())

print("SCRIPT END")