# -*- coding: utf-8 -*-
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime

# ---------- Utility parsers ----------

def parse_hms_to_seconds(s: Optional[str]) -> Optional[int]:
    """
    Convert time strings like '2:29:33', '2.29.33', '1:29:17' to total seconds.
    Returns None if s is None/empty.
    """
    if not s:
        return None
    t = s.strip().replace(".", ":")
    parts = [p for p in t.split(":") if p]
    if not parts:
        return None
    parts = [int(p) for p in parts]
    if len(parts) == 3:  # H:M:S
        h, m, s = parts
        return 3600*h + 60*m + s
    if len(parts) == 2:  # M:S
        m, s = parts
        return 60*m + s
    if len(parts) == 1:  # S
        return parts[0]
    return None

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        return float(str(x).replace(",", ""))
    except Exception:
        return None

# ---------- Schema & setup ----------

def init_db(db_path: str = "filter_tests.db") -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_name TEXT,
        date_iso TEXT,
        mains_pressure_bar REAL,

        sample_name TEXT,
        cloth_filter TEXT,

        pre_notes TEXT,
        post_notes TEXT,

        dry_solids_g REAL,
        water_g REAL,
        percent_solids REAL,

        fill_pressure_bar REAL,
        fill_time_s INTEGER,
        fill_filtrate_g REAL,

        cake_depth_mm REAL,
        cake_area_m2 REAL,
        cake_wet_weight_g REAL,
        cake_moisture_pct REAL,

        press_pressure_bar REAL,
        press_time_s INTEGER,
        press_filtrate_g REAL,

        air_blow_pressure_bar REAL,
        air_blow_time_s INTEGER,
        air_blow_filtrate_g REAL,

        solids_density_kg_m3 REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS depth_measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id INTEGER NOT NULL,
        depth_mm REAL NOT NULL,
        FOREIGN KEY(test_id) REFERENCES tests(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS moisture_measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id INTEGER NOT NULL,
        moisture_pct REAL NOT NULL,
        FOREIGN KEY(test_id) REFERENCES tests(id)
    )
    """)

    con.commit()
    con.close()

# ---------- Insert helpers ----------

def insert_test(
    db_path: str,
    record: Dict[str, Any],
    depths_mm: Optional[List[float]] = None,
    moistures_pct: Optional[List[float]] = None,
) -> int:
    """
    Insert a press-filter test. Returns the new test_id.
    'record' keys should match columns of 'tests' (see init_db()).
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cols = [c for c in record.keys()]
    vals = [record[c] for c in cols]
    placeholders = ",".join(["?"] * len(cols))

    cur.execute(f"INSERT INTO tests ({','.join(cols)}) VALUES ({placeholders})", vals)
    test_id = cur.lastrowid

    if depths_mm:
        cur.executemany(
            "INSERT INTO depth_measurements (test_id, depth_mm) VALUES (?, ?)",
            [(test_id, safe_float(d)) for d in depths_mm if safe_float(d) is not None]
        )

    if moistures_pct:
        cur.executemany(
            "INSERT INTO moisture_measurements (test_id, moisture_pct) VALUES (?, ?)",
            [(test_id, safe_float(m)) for m in moistures_pct if safe_float(m) is not None]
        )

    con.commit()
    con.close()
    return test_id

# ---------- Query helpers ----------

def get_test(db_path: str, test_id: int) -> Dict[str, Any]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA table_info(tests)")
    cols = [r[1] for r in cur.fetchall()]

    cur.execute("SELECT * FROM tests WHERE id = ?", (test_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise ValueError(f"No test with id={test_id}")

    test = dict(zip(["id"] + cols, [row[0]] + list(row[1:])))

    cur.execute("SELECT depth_mm FROM depth_measurements WHERE test_id = ?", (test_id,))
    test["depths_mm"] = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT moisture_pct FROM moisture_measurements WHERE test_id = ?", (test_id,))
    test["moistures_pct"] = [r[0] for r in cur.fetchall()]

    con.close()
    return test

# ---------- Calculations ----------

def cake_porosity(
    cake_wet_weight_g: float,
    moisture_pct: float,
    solids_density_kg_m3: float,
    area_m2: float,
    thickness_mm: float
) -> Optional[float]:
    """
    Porosity φ = 1 - [ w*(1 - m) ] / [ ρ_s * A * L ]
    w: cake wet mass [kg] (convert g → kg)
    m: moisture fraction (e.g., 0.1156 for 11.56%)
    ρ_s: solids density [kg/m3]
    A: area [m2]
    L: thickness [m] (convert mm → m)
    Returns φ (0–1), or None if inputs invalid.
    """
    try:
        w_kg = cake_wet_weight_g / 1000.0
        m = moisture_pct / 100.0
        L_m = thickness_mm / 1000.0
        denom = solids_density_kg_m3 * area_m2 * L_m
        if denom <= 0:
            return None
        phi = 1.0 - (w_kg * (1.0 - m)) / denom
        return phi
    except Exception:
        return None

def moisture_fraction(wet_g: float, dry_g: float) -> Optional[float]:
    """
    Moisture content (wet basis): (wet - dry)/wet → returns fraction (0–1).
    """
    try:
        wet = float(wet_g); dry = float(dry_g)
        if wet <= 0:
            return None
        return max(0.0, min(1.0, (wet - dry) / wet))
    except Exception:
        return None

