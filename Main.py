from PSD_tools import plot_psd, d_at_percent, plot_all_psd
from Data_Process.plot_moisture_vs_psd import plot_moisture_vs_psd_indices

def main():
    # ---- single PSD curve (as received PSD) ----
    sizes_um = [0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 26, 38, 53, 75, 106, 150, 212, 300, 425, 600, 1000]
    percent_passing = [2.826, 4.277, 7.206, 12.586, 17.421, 25.123, 31.079, 40.088, 46.615,
                       55.289, 62.971, 69.272, 75.268, 80.019, 84.453, 88.355, 91.703,
                       94.637, 97.28, 99.165, 99.914, 100]

    psd_flag = 0
    if psd_flag:
        dx = plot_psd(
            sizes_um,
            percent_passing,
            title="As Received PSD",
            x_major_ticks=(0.1, 1, 10, 100, 1000),
            x_limits=(0.1, 1000),
            annotate_dx=None,         # no D10/D50/D90 on figure
            show=True,
        )
        print("Computed D-values:", dx)

    # ---- new: plot ALL samples from PSD_Full ----
    multi_flag = 0
    if multi_flag:
        excel_path = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
        plot_all_psd(
            excel_path,
            sheet_name="PSD_Full",
            title="All PSD Curves (\u03bcm)",
            x_limits=(0.1, 1000),
            x_major_ticks=(0.1, 1, 10, 100, 1000),
            save_path="all_psd_curves.png",
            show=True,
        )

    # ---- keep your moisture/EFI/FSI analysis ----
    moisture_flag = 1
    if moisture_flag:
        data_path = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
        df_results = plot_moisture_vs_psd_indices(data_path)
        print("\nMerged DataFrame with PSD indices and Mc%:")
        print(df_results.head())

if __name__ == "__main__":
    main()
