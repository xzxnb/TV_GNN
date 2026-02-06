import typer
import mlflow
import typing as t
import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid4


def main(
    runs: t.List[str],
    xtick: t.List[int] = None,
    ytick: t.List[int] = None,
    max_x: t.Optional[float] = None,
    max_y: t.Optional[float] = None,
    min_x: t.Optional[float] = None,
    min_y: t.Optional[float] = None,
    n_steps: int = 10,
    filename: t.Optional[str] = None,
) -> None:
    client = mlflow.tracking.MlflowClient()

    plots = []
    legends = []
    min_x, max_x, min_y, max_y = (
        float("inf") if min_x is None else min_x,
        float("-inf") if max_x is None else max_x,
        float("inf") if min_y is None else min_y,
        float("-inf") if max_y is None else max_y,
    )

    for run_id in runs:
        metric = client.get_metric_history(run_id, "val/ratio_generated_in_val")
        run = client.get_run(run_id)

        xs = [m.step for m in metric][::len(metric) // n_steps]
        ys = [m.value for m in metric][::len(metric) // n_steps]

        min_x = min(min(xs), min_x)
        max_x = max(max(xs), max_x)
        min_y = min(min(ys), min_y)
        max_y = max(max(ys), max_y)

        plots.append((xs, ys))
        legends.append(run.data.tags['mlflow.runName'].split("2022 -")[1].strip())

    xtick = xtick or list(np.linspace(min_x, max_x, 10))
    ytick = ytick or list(np.linspace(min_y, max_y, 10))

    fig, ax = plt.subplots(1, 1)

    for plot, legend in zip(plots, legends):
        ax.plot(*plot, label=legend)

    ax.set_yscale("log")
    ax.legend()
    fig.savefig(f"{filename or uuid4()}.png")


if __name__ == "__main__":
    typer.run(main)
