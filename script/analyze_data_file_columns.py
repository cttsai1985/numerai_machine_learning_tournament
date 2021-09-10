import os
import json
from pathlib import Path
from typing import List
import pandas as pd

if "__main__" == __name__:
    root_dir: str = "../input/numerai_tournament_resource"

    output_dir: str = os.path.join(root_dir, "metadata")

    #
    input_file_path: str = os.path.join(
        root_dir, "latest_tournament_datasets", "numerai_validation_data.parquet")
    df = pd.read_parquet(input_file_path)

    columns: List[str] = df.columns.tolist()
    print(f"{len(columns)} columns")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    target_columns = sorted(list(filter(lambda x: x.startswith("target_"), columns)))
    output_filename: str = os.path.join(output_dir, "targets.json")
    with open(output_filename, "w") as f:
        json.dump(target_columns, f)
        print(f"save {len(target_columns)} targets to {output_filename}")

    feature_columns = sorted(list(filter(lambda x: x.startswith("feature_"), columns)))
    output_filename: str = os.path.join(output_dir, "features_numerai.json")
    with open(output_filename, "w") as f:
        json.dump(feature_columns, f)
        print(f"save {len(feature_columns)} features to {output_filename}")
