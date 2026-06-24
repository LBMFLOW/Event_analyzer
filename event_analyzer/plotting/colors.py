from __future__ import annotations


PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def color_for_index(index: int) -> str:
    return PALETTE[index % len(PALETTE)]

