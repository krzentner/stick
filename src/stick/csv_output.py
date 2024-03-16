import datetime
import re
import os
from typing import Optional, Union
import csv

import stick


@stick.declare_output_engine
class CSVOutputEngine(stick.OutputEngine):
    def __init__(
        self,
        log_dir: stick.utils.FileIsh,
        run_name: str,
        log_level=stick.TRACE,
    ):
        super().__init__(log_level=log_level)
        self.log_dir = os.path.abspath(log_dir)
        self.run_name = run_name
        self.writers = {}

    def log_row_inner(self, row: stick.Row):
        msg = row.as_flat_dict()
        msg["$step"] = row.step
        msg["$utc_timestamp"] = datetime.datetime.utcnow().timestamp()
        msg["$level"] = int(row.log_level)
        msg = dict(sorted(msg.items()))

        if row.table_name in self.writers:
            f, writer = self.writers[row.table_name]
        else:
            f_name = os.path.join(self.log_dir, self.run_name, f"{row.table_name}.csv")
            f = stick.utils.FileManager(f_name)
            f.should_close = True
            writer = csv.DictWriter(f.file, fieldnames=msg.keys())
            writer.writeheader()
            self.writers[row.table_name] = (f, writer)
        writer.writerow(msg)
        f.file.flush()

    def close(self):
        for f, writers in self.writers.values():
            f.close()


def _try_convert(s: str) -> Union[str, float]:
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_csv_file(
    filename: str, keys: Optional[list[str]] = None
) -> dict[str, list[stick.flat_utils.ScalarTypes]]:
    with open(filename) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if keys is None:
        keys = list(rows[0].keys())
    return {k: [_try_convert(row[k]) for row in rows] for k in keys}


stick.LOAD_FILETYPES[".csv"] = load_csv_file
