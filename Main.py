# -*- coding: utf-8 -*-

from PSD_tools import plot_psd, d_at_percent, plot_all_psd
from Data_Process.plot_moisture_vs_psd import plot_moisture_vs_psd_indices
from Data_Process.T_vs_Ml import plot_T_vs_Ml, plot_TvsPSD
from Data_Process.Pumping_curve import plot_pumping_curve
from Data_Process.PSDvsCakePore import plot_PSD_vs_CakePore

def main():
    # === Path to your live Excel file ===
    data_path = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"

    # === Define which samples to include globally ===
    include_samples = ["Si_C", "Si_M"]
    #include_samples = None  # <- to plot all

    # === Global consistent colors ===
    SAMPLE_COLORS = {
        "Si_F":   "#1f77b4",   # blue
        "Si_M":   "#ff7f0e",   # orange
        "Si_C":   "#2ca02c",   # green
        "Si_Rep": "#d62728",   # red
        "Si_Rep_new": "#9467bd",  # purple
        # add others as needed
    }
    color_map = SAMPLE_COLORS  # <- alias used below

    # === Flags ===
    psd_flag       = 0
    multi_flag     = 1
    moisture_flag  = 1
    tvml_flag      = 0
    tvspsd_flag    = 1
    pump_flag      = 0
    cake_pore_flag = 1

    # ---- Single PSD curve ----
    if psd_flag:
        sizes_um = [0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 26, 38, 53, 75, 106, 150, 212, 300, 425, 600, 1000]
        percent_passing = [2.826, 4.277, 7.206, 12.586, 17.421, 25.123, 31.079, 40.088, 46.615,
                           55.289, 62.971, 69.272, 75.268, 80.019, 84.453, 88.355, 91.703,
                           94.637, 97.28, 99.165, 99.914, 100]

        dx = plot_psd(
            sizes_um,
            percent_passing,
            title="As Received PSD",
            x_major_ticks=(0.1, 1, 10, 100, 1000),
            x_limits=(0.1, 1000),
            annotate_dx=None,
            show=True,
            label="Si_F",
            color_map=SAMPLE_COLORS
        )       

    # ---- Multiple PSDs ----
    if multi_flag:
        plot_all_psd(
            data_path,
            sheet_name="PSD_Full",
            include_samples=include_samples,  # or None
            color_map=SAMPLE_COLORS,
            show=True
        )

    # ---- Mc% vs EFI / FSI ----
    if moisture_flag:
        df_results = plot_moisture_vs_psd_indices(
            data_path,
            include_samples=include_samples,
            color_map=color_map
        )
        print("\nMerged DataFrame with PSD indices and Mc%:")
        print(df_results.head())

    # ---- F_T vs F_V/F_T ----
    if tvml_flag:
        df_tvml = plot_T_vs_Ml(
            data_path,
            include_samples=include_samples,
            color_map=color_map
        )
        print("\nFiltered DataFrame for F_T vs F_V/F_T:")
        print(df_tvml[["Sample Code", "F_T", "F_V", "FT_over_FV"]].head())

    # ---- F_T/F_V vs PSD index ----
    if tvspsd_flag:
        df_tvspsd = plot_TvsPSD(
            data_path,
            include_samples=include_samples,
            color_map=color_map
        )
        print(df_tvspsd[["Sample Code", "EFI", "FT_over_FV"]].head())

    # ---- Pumping curve ----
    if pump_flag:
        plot_pumping_curve()

    # ---- Cake Porosity vs PSD ----
    if cake_pore_flag:
        try:
            plot_PSD_vs_CakePore(
                data_path,
                sheet_db="DB",
                sheet_psd="PSD",
                d50_col="D50",
                include_samples=include_samples,
                color_map=color_map
            )
        except Exception as e:
            print("CakePore plot failed:", e)

if __name__ == "__main__":
    main()
