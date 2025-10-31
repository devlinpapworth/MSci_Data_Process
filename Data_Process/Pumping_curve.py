
"""
pumping_curve.py
-----------------
Plot a Pumping Curve (time vs. pressure) from hard-coded data or a provided list.
Usage:
    python pumping_curve.py  # uses built-in data
Or import:
    from pumping_curve import plot_pumping_curve
"""

from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt

# Default dataset: (label, time_sec, pressure_bar)
DEFAULT_POINTS = [
    ("Recirulation", -30, 2.25),
    ("Recirulation", -25, 2.25),
    ("Recirulation", -20, 2.25),
    ("Recirulation", -15, 2.25),
    ("Recirulation", -10, 2.25),
    ("Recirulation", -5,  2.25),
    ("start",         0,  2.25),
    ("filling",       3.0, 2.25),
    ("filling",       7.0, 2.25),
    ("filling",      10.0, 3.25),
    ("filling",      12.0, 3.25),
    ("filling",      15.0, 4.0),
    ("filling",      17.0, 4.0),
    ("filling",      20.0, 4.0),
    ("filling",      25.0, 4.0),
    ("filling",      30.0, 4.0),
    ("filling",      40.0, 4.0),
    ("filling",      50.0, 4.0),
    ("filling",      52.2, 4.0),
]

def plot_pumping_curve(
    points: Optional[Iterable[Tuple[str, float, float]]] = None,
    save_path: Optional[str] = "pumping_curve.png",
    show: bool = True,
) -> None:
    """
    Plot time (s) vs pressure (bar).

    Args:
        points: iterable of (label, time_s, pressure_bar). If None, uses DEFAULT_POINTS.
        save_path: where to save the PNG; pass None to skip saving.
        show: whether to display the chart interactively.
    """
    if points is None:
        points = DEFAULT_POINTS

    # Sort by time so the line draws correctly
    points = sorted(points, key=lambda r: r[1])

    times = [p[1] for p in points]
    pressures = [p[2] for p in points]

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, pressures, marker="o")  # no explicit colors per instruction
    plt.title("Pumping Curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (bar)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # Mark t=0
    if 0 >= min(times) and 0 <= max(times):
        plt.axvline(0, linestyle=":", linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

def main():
    plot_pumping_curve()

if __name__ == "__main__":
    main()

