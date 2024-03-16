import numpy as np

import stick
from stick.csv_output import CSVOutputEngine
import stick.np


def test_summarize_array(tmp_path):
    run_name = "test_summarize_array"
    output = CSVOutputEngine(log_dir=tmp_path, run_name=run_name)
    output.log_row_inner(
        stick.Row(
            table_name="test_table1",
            raw={"a": np.arange(10)},
            step=10,
        )
    )
    f_name = f"{tmp_path}/{run_name}/test_table1.csv"
    data = stick.csv_output.load_csv_file(f_name)
    assert data["a.mean"] == [4.5]
    assert data["a.min"] == [0]
    assert data["a.max"] == [9]
    assert data["a.std"][0] > 1
