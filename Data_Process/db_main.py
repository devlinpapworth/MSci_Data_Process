
# -*- coding: utf-8 -*-
from filter_db import (
    init_db, insert_test, get_test,
    parse_hms_to_seconds, cake_porosity
)

def main():
    db_path = "filter_tests.db"
    init_db(db_path)

    # ---- Fill this dict from your sheet ----
    record = {
        "test_name": "Silica new rep @ 130",
        "date_iso": "2025-10-28",           # ISO yyyy-mm-dd
        "mains_pressure_bar": 7.0,

        "sample_name": "silica representative",
        "cloth_filter": "130",

        "pre_notes": "",
        "post_notes": "filtrate semi clear; chamber fully filled",

        "dry_solids_g": 1349.3,
        "water_g": 1359.6,
        "percent_solids": 49.81,

        # Filling
        "fill_pressure_bar": 3.25,
        "fill_time_s": parse_hms_to_seconds("2.29.33"),  # hh.mm.ss or hh:mm:ss
        "fill_filtrate_g": 513.3,

        # Cake
        "cake_depth_mm": 45.0,
        "cake_area_m2": 0.01,
        "cake_wet_weight_g": 790.3,
        "cake_moisture_pct": 11.56,  # if you trust the sheet value

        # Pressing (unknowns left None)
        "press_pressure_bar": None,
        "press_time_s": None,
        "press_filtrate_g": None,

        # Air blow
        "air_blow_pressure_bar": 5.0,
        "air_blow_time_s": parse_hms_to_seconds("1.29.17"),
        "air_blow_filtrate_g": 150.3,

        "solids_density_kg_m3": 2648.0,
    }

    # Depth replicated measurements (you had five 45 mm entries)
    depths_mm = [45, 45, 45, 45, 45]

    # Moisture replicate list (you had only one value listed)
    moistures_pct = [11.56]

    # ---- Insert into DB ----
    test_id = insert_test(db_path, record, depths_mm=depths_mm, moistures_pct=moistures_pct)
    print(f"Saved test_id = {test_id}")

    # ---- Read back & compute ----
    saved = get_test(db_path, test_id)
    print("Loaded test:", {k: saved[k] for k in ("id", "test_name", "date_iso")})

    # Example calculation: cake porosity from saved values
    phi = cake_porosity(
        cake_wet_weight_g=saved["cake_wet_weight_g"],
        moisture_pct=saved["cake_moisture_pct"],
        solids_density_kg_m3=saved["solids_density_kg_m3"],
        area_m2=saved["cake_area_m2"],
        thickness_mm=saved["cake_depth_mm"],
    )
    if phi is not None:
        print(f"Cake porosity (calc): {phi:.6f} (fraction)")
    else:
        print("Cake porosity could not be computed (missing/invalid inputs).")

    # Example: average depth from replicates
    if saved["depths_mm"]:
        avg_depth = sum(saved["depths_mm"]) / len(saved["depths_mm"])
        print(f"Average measured depth: {avg_depth:.2f} mm")

if __name__ == "__main__":
    main()
