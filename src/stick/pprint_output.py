"""Output by using python's builtin pprint. Mostly suitable for debug logs."""
from pprint import pprint

import stick
from stick import OutputEngine, declare_output_engine


@declare_output_engine
class PPrintOutputEngine(OutputEngine):
    def __init__(
        self,
        file: stick._utils.FileIsh = None,
        flatten: bool = True,
        log_level=stick.RESULTS,
    ):
        super().__init__(log_level=log_level)
        self.fm = stick._utils.FileManager(file)
        self.flatten = flatten

    def log_row_inner(self, row):
        print(f"Table {row.table_name}:", file=self.fm.file)
        if self.flatten:
            msg = row.as_summary()
        else:
            msg = row.raw
        pprint(msg, stream=self.fm.file)

    def close(self):
        self.fm.close()
