import pandas as pd
from io import StringIO

with open ("measure.txt", "r") as f:
    data = f.read()

for file in data.split("\n\n#"):
    colnames = "index, note, duration, tactus, joint"
    file_csv = StringIO(colnames + "\n" + "\n".join(file.splitlines()[1:]))
    filename = file.splitlines()[0][1:].replace("\\", "")[:-4].replace(" ", "")
    df = pd.read_csv(file_csv)
    df.to_csv(f"preprocessed/{filename}.csv", index=False, encoding="utf8")
