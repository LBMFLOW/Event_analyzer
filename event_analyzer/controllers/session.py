from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


@dataclass(slots=True)
class SessionState:
    csv_path: str = ""
    time_column: str = ""
    target_columns: list[str] = field(default_factory=list)
    auxiliary_columns: list[str] = field(default_factory=list)
    auxiliary_axes: dict[str, str] = field(default_factory=dict)
    header_row: int = 1
    units_row: int | None = None
    data_start_row: int = 2
    plot_title: str = "Time-series plot"
    x_axis_title: str = ""
    y_axis_title: str = ""
    dividers: list[dict[str, object]] = field(default_factory=list)
    threshold: float | None = None
    region: tuple[float, float] | None = None
    region_name: str = ""
    colors: dict[str, str] = field(default_factory=dict)
    visibility: dict[str, bool] = field(default_factory=dict)
    theme: str = "light"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SessionState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        region = data.get("region")
        if region is not None:
            data["region"] = (float(region[0]), float(region[1]))
        if data.get("dividers") and isinstance(data["dividers"][0], (int, float)):
            data["dividers"] = [{"time": float(value)} for value in data["dividers"]]
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


__all__ = ["SessionState"]
