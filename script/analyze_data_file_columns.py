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

    # parse all feature in data
    target_columns = sorted(list(filter(lambda x: x.startswith("target_"), columns)))
    output_filename: str = os.path.join(output_dir, ft.default_target_collection_filename)
    with open(output_filename, "w") as f:
        json.dump(target_columns, f, sort_keys=True, indent=4)
        print(f"save {len(target_columns)} targets to {output_filename}")

    feature_columns = sorted(list(filter(lambda x: x.startswith("feature_"), columns)))
    output_filename: str = os.path.join(output_dir, ft.default_feature_collection_filename)
    with open(output_filename, "w") as f:
        json.dump(feature_columns, f)
        print(f"save {len(feature_columns)} features to {output_filename}")

    # numer ai selection
    default_feature_stats_filename: str = os.path.join(ft.default_data_dir, "features.json")
    with open(default_feature_stats_filename, 'r') as f:
        obj = json.load(f)

    for feature_set_name, feature_columns in obj['feature_sets'].items():
        output_filename: str = os.path.join(output_dir, f"features_{feature_set_name}.json")
        with open(output_filename, "w") as f:
            json.dump(feature_columns, f, sort_keys=True, indent=4)
            print(f"save {len(feature_columns)} features to {output_filename}")
