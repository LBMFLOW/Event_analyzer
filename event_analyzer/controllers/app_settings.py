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
    last_csv_directory: str = ""
    last_save_directory: str = ""

    def add_recent_file(self, path: str | Path) -> None:
        resolved = str(Path(path))
        remaining = [item for item in self.recent_files if item != resolved]
        self.recent_files = [resolved, *remaining][:MAX_RECENT_FILES]
        directory = _csv_parent_directory(resolved)
        if directory:
            self.last_csv_directory = directory

    def open_csv_directory(self) -> str:
        """Return the best available folder to use for the Open CSV dialog."""
        if self.last_csv_directory:
            directory = Path(self.last_csv_directory).expanduser()
            if directory.is_dir():
                return str(directory)
        for recent_file in self.recent_files:
            directory = Path(recent_file).expanduser().parent
            if directory.is_dir():
                return str(directory)
        return ""

    def remember_save_path(self, path: str | Path) -> None:
        """Remember the folder used by any Save/Export dialog."""
        directory = _parent_directory(path)
        if directory:
            self.last_save_directory = directory

    def save_directory(self) -> str:
        """Return the best available folder to use for Save/Export dialogs."""
        if self.last_save_directory:
            directory = Path(self.last_save_directory).expanduser()
            if directory.is_dir():
                return str(directory)
        csv_directory = self.open_csv_directory()
        if csv_directory:
            return csv_directory
        return ""

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
        last_csv_directory = str(data.get("last_csv_directory", "") or "")
        last_save_directory = str(data.get("last_save_directory", "") or "")
        return cls(
            recent_files=recent[:MAX_RECENT_FILES],
            theme=theme,
            last_csv_directory=last_csv_directory,
            last_save_directory=last_save_directory,
        )


def default_settings_path() -> Path:
    """Return the user preference file path without hardcoding a platform path."""
    return Path.home() / ".event_analyzer" / "settings.json"


def _csv_parent_directory(path: str | Path) -> str:
    return _parent_directory(path)


def _parent_directory(path: str | Path) -> str:
    parent = Path(path).expanduser().parent
    if str(parent) in {"", "."}:
        return ""
    return str(parent)


__all__ = ["AppSettings", "MAX_RECENT_FILES", "default_settings_path"]
