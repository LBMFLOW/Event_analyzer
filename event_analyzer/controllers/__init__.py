"""Controller package."""

__all__ = [
    "Divider",
    "DividerManager",
    "DividerStyle",
    "RegionSelector",
    "SelectedRegion",
    "ThresholdManager",
    "ThresholdState",
    "ThresholdSummary",
]


def __getattr__(name: str):
    if name in {"Divider", "DividerManager", "DividerStyle"}:
        from event_analyzer.controllers.divider_manager import Divider, DividerManager, DividerStyle

        return {"Divider": Divider, "DividerManager": DividerManager, "DividerStyle": DividerStyle}[name]
    if name in {"RegionSelector", "SelectedRegion"}:
        from event_analyzer.controllers.region_selector import RegionSelector, SelectedRegion

        return {"RegionSelector": RegionSelector, "SelectedRegion": SelectedRegion}[name]
    if name in {"ThresholdManager", "ThresholdState", "ThresholdSummary"}:
        from event_analyzer.controllers.threshold_manager import ThresholdManager, ThresholdState, ThresholdSummary

        return {
            "ThresholdManager": ThresholdManager,
            "ThresholdState": ThresholdState,
            "ThresholdSummary": ThresholdSummary,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
