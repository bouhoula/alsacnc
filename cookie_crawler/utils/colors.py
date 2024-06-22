import re
from typing import Dict, List, Tuple

from shared_utils import read_txt_file


class ColorDB:
    def __init__(self, rgb_filename: str = "config/rgb.txt"):
        self.name_to_rgb: Dict[str, Tuple[int, ...]] = dict()
        self.rgb_to_name: Dict[Tuple[int, ...], Tuple[str, List]] = dict()
        pattern = re.compile(
            r"\s*(?P<red>\d+)\s+(?P<green>\d+)\s+(?P<blue>\d+)\s+(?P<name>.*)"
        )
        lines = read_txt_file(rgb_filename)
        for line in lines:
            mo = pattern.match(line)
            if not mo:
                continue
            rgb = tuple(int(x) for x in mo.group("red", "green", "blue"))
            name = mo.group("name")
            keyname = name.lower()
            name_found, aliases = self.rgb_to_name.get(rgb, (name, []))
            if name_found != name and name_found not in aliases:
                aliases.append(name)
            self.rgb_to_name[rgb] = (name_found, aliases)
            self.name_to_rgb[keyname] = rgb

    def find_nearest_color(self, red: int, green: int, blue: int) -> str:
        nearest = -1
        nearest_name = ""
        for name, aliases in self.rgb_to_name.values():
            r, g, b = self.name_to_rgb[name.lower()]
            rdelta = red - r
            gdelta = green - g
            bdelta = blue - b
            distance = rdelta * rdelta + gdelta * gdelta + bdelta * bdelta
            if nearest == -1 or distance < nearest:
                nearest = distance
                nearest_name = name
        return nearest_name


if __name__ == "__main__":
    color_db = ColorDB()
