from PSD_tools import plot_psd
#from Data_Process import 
def main():
    # Example data – replace with yours
    sizes_um = [0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 26, 38, 53, 75, 106, 150, 212, 300, 425, 600, 1000]
    percent_passing = [2.826, 4.277, 7.206, 12.586, 17.421, 25.123, 31.079, 40.088, 46.615, 55.289, 62.971, 69.272, 75.268, 80.019, 84.453, 88.355, 91.703, 94.637, 97.28, 99.165, 99.914, 100]

    psd = 1   # flag to toggle plotting on/off

    if psd == 1:
        dx = plot_psd(
            sizes_um,
            percent_passing,
            title="As Received PSD",
            x_major_ticks=(0.1, 1, 10, 100, 1000),
            x_limits=(0.1, 1000),
            annotate_dx=(10, 50, 90),
            show=True,
        )
        print("Computed D-values:", dx)


if __name__ == "__main__":
    main()
