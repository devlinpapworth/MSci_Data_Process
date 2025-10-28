
from PSD_tools import plot_psd

def main():
    # Example data – replace with yours
    sizes_um = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    percent_passing = [1, 3, 10, 20, 35, 55, 70, 82, 93, 99]

    dx = plot_psd(
        sizes_um,
        percent_passing,
        title="Sample XYZ – PSD",
        x_major_ticks=(0.1, 1, 10, 100),   # your requested ticks
        x_limits=(0.1, 100),
        annotate_dx=(10, 50, 90),
        save_path="psd_sample_xyz.png",
        show=True,
    )

    print("Computed D-values:", dx)

if __name__ == "__main__":
    main()
