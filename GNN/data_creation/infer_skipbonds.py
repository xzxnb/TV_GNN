import typer

from pathlib import Path
from pyswip import Prolog


def main(
    filename: str,
    read_folder: Path,
    write_folder: Path,
    add_skipbonds: bool,
):
    if add_skipbonds:
        prolog = Prolog()

        prolog.consult(str(read_folder / filename))
        prolog.consult("data_creation/skipBond.pl")

        results = [
            "skipBond(%s,%s).\n" % (t["X"], t["Y"])
            for t in prolog.query("skipBond(X,Y).")
        ]
    else:
        results = []

    with open(read_folder / filename) as f_read:
        for line in f_read:
            results.insert(0, line)

    with open(write_folder / filename, "w") as f_write:
        for result in results:
            f_write.write(result)


if __name__ == "__main__":
    typer.run(main)
