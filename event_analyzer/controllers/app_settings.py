from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


MAX_RECENT_FILES = 10


@dataclass(slots=True)
class AppSettings:
    """Small cross-session preference store."""

    recent_files: list[str] = field(default_factory=list)
    theme: str = "light"

    def add_recent_file(self, path: str | Path) -> None:
        resolved = str(Path(path))
        remaining = [item for item in self.recent_files if item != resolved]
        self.recent_files = [resolved, *remaining][:MAX_RECENT_FILES]

    def save(self, path: str | Path | None = None) -> None:
        settings_path = Path(path) if path is not None else default_settings_path()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppSettings":
        settings_path = Path(path) if path is not None else default_settings_path()
        if not settings_path.exists():
            return cls()
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        recent = [str(item) for item in data.get("recent_files", []) if item]
        theme = str(data.get("theme", "light"))
        if theme not in {"light", "dark"}:
            theme = "light"
        return cls(recent_files=recent[:MAX_RECENT_FILES], theme=theme)


def default_settings_path() -> Path:
    """Return the user preference file path without hardcoding a platform path."""
    return Path.home() / ".event_analyzer" / "settings.json"


__all__ = ["AppSettings", "MAX_RECENT_FILES", "default_settings_path"]
