import os
import json
from pathlib import Path
from typing import List
import pandas as pd

from ds_utils import FilenameTemplate as ft

if "__main__" == __name__:
    root_dir: str = ft.root_resource_path
    output_dir: str = ft.default_meta_data_dir

    #
    input_file_path: str = os.path.join(
        ft.default_data_dir, ft.numerai_data_filename_template.format(eval_type="validation"))
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
