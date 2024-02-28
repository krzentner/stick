import numpy as np
import stick
import argparse
import visdom
from pathlib import Path

X_RES = 500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    return parser.parse_args()


def main(args):
    vis = visdom.Visdom()
    log_dir = Path(args.log_dir)
    log_data = stick.load_log_file(log_dir / "stick.ndjson")
    print([k for k in log_data.keys() if k.startswith("$step")])
    for k, v in log_data.items():
        if not isinstance(v[0], str) and not k.startswith("$"):
            if k.count(".") <= 1:
                table = k.split(".")[0]
                print(f"Plotting {k}")
                try:
                    x_axis = log_data[f"$step.{table}"]
                except KeyError:
                    print(f"Could not get x axis for {k}")
                else:
                    if len(x_axis) > X_RES:
                        print(f"Only plotting last {X_RES} points of {k}")
                        x_axis = x_axis[-X_RES:]
                        v = v[-X_RES:]
                    if len(x_axis) == len(v):
                        vis.line(
                            X=x_axis,
                            Y=v,
                            opts=dict(
                                title=k,
                            ),
                        )


if __name__ == "__main__":
    main(parse_args())
