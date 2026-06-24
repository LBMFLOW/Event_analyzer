"""Data package.

Import concrete classes from their modules. Keeping this package initializer
lightweight lets the pure analysis unit tests run without GUI or CSV optional
dependencies already installed.
"""

__all__ = [
    "CSVMetadata",
    "CSVPreview",
    "ColumnProfile",
    "DataManager",
    "DataManagerError",
    "EmptyDataError",
    "InvalidTimeColumnError",
    "LoadedData",
    "MissingFileError",
    "NonNumericColumnError",
    "TooManyInvalidValuesError",
    "UnreadableCSVError",
]


def __getattr__(name: str):
    if name in __all__:
        from event_analyzer.data import data_manager

        return getattr(data_manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
