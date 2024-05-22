Because a package called "stick" already exists on pypi, I have renamed `stick` to `noko`. You can read me on the [github noko page](https://github.com/krzentner/noko), or run:

```sh
pip install "noko[recommended]"
```


# stick

[\[api reference\]](https://krzentner.github.io/stick/)

Log first, ask questions later.

Stick is a numerical logging library designed for logging as many variables at once on as many timescales as possible.

The prototypical use case for stick is the following:

```python
loss.backward()
stick.log_row("grad_step", locals(), level=stick.TRACE)
optimizer.step()
```

Which would produce a log file "grad_step.csv", with information on all local variables at every gradient step.

## State of the Library

Stick is quite useful already, but I'm continuing to refine the API, add new features as I find need of them, and fix bugs as I encounter them.

If you use the library at this time, you should expect API churn,
so depend on a specific version.

The current recommended install method is:

```
pip install "git+https://github.com/krzentner/stick.git@v0.1.1#egg=stick[recommended]"
```


## Key Idea

Most numerical logging libraries expect you to log one key at a time.
Stick instead expects you to log one "row" at a time, into a named "table". In the above example, the table is named "grad_step", and the row is all local variables.

Stick will then go through all values in the row, and determine how to summarize them.
For example, it will summarize a pytorch neural network using the min, max, mean, and std of each parameter, as well as those values for the gradient of each parameter (if available).

Once summarized, stick outputs that row to all stick backends that have a log level at least as sensitive as the provided level.

## Backends

Always available:
  - newline delimited json (.ndjson)
  - comma separated values (.csv)
  - pprint (usually only used for stdout)

Requires optional dependencies:
  - tensorboard (will use torch.utils.tensorboard or tensorboardX or tf.summary)
  - parquet (requires pyarrow)

## Row Summarization

Stick runs a preprocessing step on each provided row before logging it.
This pre-process is typically *lossy*, and summarizes large input arrays into a few scalars.
This is necessary to be able to log all local variables, and is an unavoidable trade-off of stick's design.
If you want to be able to exactly recover a value passed to stick, format it to a `str` or `bytes` first, and use the csv, ndjson, or parquet backends.

For built-in datatypes (dictionaries, lists, tuples), preprocessing runs recursively.
For commonly used numerical libraries (numpy, pytorch), there are adapter modules to summarize common datatypes.

If you want your type to be flattened, you can define a custom summarizer. The standard way to do this is to use `declare_summarizer`, which can be used without modifying the source for the particular type being summarized.

Example of summarizing a custom type "MyPoint":

```python
from stick import declare_summarizer, summarize, ScalarTypes

@declare_summarizer(MyPoint):
def summarize_mypoint(point, prefix: str, dst: dict[str, ScalarTypes]):
    # Include x and y fields directly
    dst[f"{prefix}.x"] = point.x
    dst[f"{prefix}.y"] = point.y
    # Recursively summarize metadata
    summarize(point.metadata, f"{prefix}.metadata", dst)
```

Sometimes you may have local variables that don't make sense to log in stick (e.g. long lists).
In that case the recommended pattern is:

```python
row = locals()
del row["my_long_list"]
stick.log_row("my_locals", row)
```

## Extra Utilities

Stick has some other useful tricks for reproducibility which can be accessed via calling `stick.init_extra()`.

These include:
  - wandb sync and config logging (if `wandb` is installed).
  - automatically setting the global seed values for most
    libraries if present in the config
  - automatically creating a git commit of the current code on
    a separate branch `stick-checkpoints`, and adding diffs
    using that checkpoint to the log directory (requires `GitPython` to be installed).
