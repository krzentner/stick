import datetime
import re
import os
from typing import Optional

import pyarrow as pa
import pyarrow.fs
import pyarrow.csv
import pyarrow.parquet

import stick

ALL_FILETYPES = [".csv", ".parquet"]
DEFAULT_FILETYPES = [".csv", ".parquet"]


@stick.declare_output_engine
class ArrowOutputEngine(stick.OutputEngine):
    def __init__(
        self,
        log_dir: stick.utils.FileIsh,
        run_name: str,
        filetypes=DEFAULT_FILETYPES,
        log_level=stick.TRACE,
    ):
        # The arrow backend is relatively efficient, so default to TRACE level.

        super().__init__(log_level=log_level)

        # Convert non-uri paths to absolute paths
        if not re.match(r"^[a-z0-9]+://", log_dir):
            log_dir = os.path.abspath(log_dir)

        self.base_uri = log_dir

        self.fs, self.base_path = pa.fs.FileSystem.from_uri(log_dir)
        self.fs.create_dir(self.base_path)
        self.filetypes = filetypes
        self.run_name = run_name

        self.writers = {}

    def log_row_inner(self, row):
        msg = row.as_flat_dict()
        msg["$step"] = row.step
        msg["$utc_timestamp"] = datetime.datetime.utcnow().timestamp()
        msg["$level"] = int(row.log_level)

        record = pa.RecordBatch.from_pylist([msg])

        schema = record.schema

        if row.table_name not in self.writers:
            self.writers[row.table_name] = {
                ext: self._create_writer(row.table_name, ext, schema)
                for ext in self.filetypes
            }

        writers = self.writers[row.table_name]
        for writer in writers.values():
            try:
                writer.write(record)
            except ValueError:
                # Probably a schema mis-match
                error_msgs = []
                for i in range(min(len(writer.schema), len(record.schema))):
                    if writer.schema[i] != record.schema[i]:
                        error_msgs.append(
                            f"Schema mismatch at index {i}: "
                            f"{writer.schema[i]} vs {record.schema[i]}"
                        )
                raise ValueError("\n".join(error_msgs))

    def close(self):
        for writers in self.writers.values():
            for writer in writers.values():
                writer.close()

    def _create_writer(self, table: str, ext: str, schema: pa.Schema):
        path = f"{self.base_path}/{self.run_name}/{table}{ext}"
        # print("creating writer", path)
        stream = self.fs.open_output_stream(path)
        if ext == ".csv":
            return pa.csv.CSVWriter(stream, schema)
        elif ext == ".parquet":
            return pa.parquet.ParquetWriter(stream, schema)


def load_parquet_file(
    filename: str, keys: Optional[list[str]]
) -> dict[str, list[stick.flat_utils.ScalarTypes]]:
    table = pa.parquet.read_table(filename, columns=keys)
    return table.to_pydict()


stick.LOAD_FILETYPES[".parquet"] = load_parquet_file


def load_csv_file(
    filename: str, keys: Optional[list[str]]
) -> dict[str, list[stick.flat_utils.ScalarTypes]]:
    table = pa.csv.read_csv(
        filename, read_options=pa.csv.ReadOptions(column_names=keys)
    )
    return table.to_pydict()


stick.LOAD_FILETYPES[".csv"] = load_csv_file
