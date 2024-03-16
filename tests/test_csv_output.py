import csv

import stick
from stick.csv_output import CSVOutputEngine


def test_write(tmp_path):
    run_name = "test_run"
    output = CSVOutputEngine(log_dir=tmp_path, run_name=run_name)
    output.log_row_inner(
        stick.Row(
            table_name="test_table1",
            raw={"y": 0, "x": 1},
            step=10,
        )
    )
    output.log_row_inner(
        stick.Row(
            table_name="test_table1",
            raw={
                "x": 1,
                "y": "3",
            },
            step=20,
        )
    )
    with open(f"{tmp_path}/{run_name}/test_table1.csv") as f:
        content = f.read()
    lines = content.split("\n")
    assert lines[0] == "$level,$step,$utc_timestamp,x,y"
    assert lines[1].endswith("1,0")
    assert lines[2].endswith("1,3")
    # There's one trailing newline:
    assert lines[3] == ""
    assert len(lines) == 4


def test_read(tmp_path):
    f_name = f"{tmp_path}/test.csv"
    with open(f_name, "w") as f:
        w = csv.DictWriter(f, ["a", "b"])
        w.writeheader()
        w.writerow({"a": 0, "b": "test1"})
        w.writerow({"a": 1, "b": "test2"})
        w.writerow({"a": None, "b": 3.5})
    data = stick.csv_output.load_csv_file(f_name)
    assert sorted(data.keys()) == ["a", "b"]
    assert data["a"] == [0, 1, None]
    assert data["b"] == ["test1", "test2", 3.5]


def test_write_inconsistent_keys(tmp_path):
    run_name = "test_run"
    output = CSVOutputEngine(log_dir=tmp_path, run_name=run_name)
    output.log_row_inner(
        stick.Row(
            table_name="test_table1",
            raw={"y": 0, "x": 1},
            step=10,
        )
    )
    output.log_row_inner(
        stick.Row(
            table_name="test_table1",
            raw={"y": 0, "x": 2, "z": 10},
            step=20,
        )
    )
    output.log_row_inner(
        stick.Row(
            table_name="test_table1",
            raw={"x": 3, "z": 20},
            step=20,
        )
    )
    f_name = f"{tmp_path}/{run_name}/test_table1.csv"
    data = stick.csv_output.load_csv_file(f_name)
    data["x"] == [1, 2, 3]
    data["y"] == [0, 0, None]
    data["z"] == [None, 10, 20]
