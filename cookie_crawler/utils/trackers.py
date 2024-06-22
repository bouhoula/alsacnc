import json
from pathlib import Path
from typing import Dict, Optional

import click


def get_most_prevalent_trackers(
    trackers_dir: str,
    prevalence_threshold: float,
    output_filename: Optional[str] = None,
) -> Dict[str, Dict]:
    trackers_filenames = (Path(trackers_dir) / "entities").glob("*")
    prevalent_trackers: Dict[str, Dict] = dict()
    for filename in trackers_filenames:
        with open(filename, "r") as fin:
            tracker_info = json.load(fin)
        if (
            "prevalence" not in tracker_info
            or tracker_info["prevalence"]["tracking"] < prevalence_threshold
        ):
            continue
        prevalent_trackers[tracker_info["displayName"]] = {
            "properties": tracker_info["properties"],
            "prevalence": tracker_info["prevalence"]["tracking"],
        }
    if output_filename is not None:
        with open(output_filename, "w") as fout:
            json.dump(prevalent_trackers, fout)
    return prevalent_trackers


@click.command()
@click.option("--tracker_radar_dir")
@click.option("--prevalence_threshold", default=0.05)
@click.option("--output_filename", default="trackers.json")
def main(
    tracker_radar_dir: str, prevalence_threshold: float, output_filename: str
) -> None:
    prevalent_trackers = get_most_prevalent_trackers(
        tracker_radar_dir, prevalence_threshold, output_filename
    )
    print(len(prevalent_trackers))


if __name__ == "__main__":
    main()
