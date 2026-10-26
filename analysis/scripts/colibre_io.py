"""Common input/output for the COLIBRE analysis drivers: command line, snapshot loading,
output folders and figure saving."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import swiftsimio as sw


DEFAULT_SNAP_DIR = "/cosma8/data/dp004/colibre/Runs"
DEFAULT_SAVE_DIR = "/cosma8/data/do019/dc-fulg1/outputs/swiftsimio_analysis"

# One sub-folder per topic, numbered in the order of the analysis
FIG_DIRS = {
    "gas"        : "01_Gas",                                   # full-box maps, phase diagrams
    "HI"         : "02_HI",                                    # slab HI maps, N_HI distribution
    "metal_maps" : os.path.join("03_Metals", "Maps"),          # slab column and abundance maps
    "metal_diag" : os.path.join("03_Metals", "Diagnostics"),   # distributions, absorber classes
    "HI_models"  : os.path.join("04_Ions", "HI_models"),       # equilibrium HI models vs CHIMES (validation)
    "metal_ions" : os.path.join("04_Ions", "Metal_ions"),      # metal ion fractions (prediction)
}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(description="Analyse a COLIBRE snapshot for JAGGER"):
    """Command-line options: which run/snapshot to analyse and where to save."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run", default="L0025N0188/Thermal",   # default light
                   help="run directory relative to --snap-dir (e.g. L0025N0188/Thermal)")
    p.add_argument("--snap", type=int, required=True,
                   help="snapshot number (L025_m5: 44 z=5.5, 48 z=5.0, 52 z=4.5, 56 z=4.0)")
    p.add_argument("--snap-dir", default=DEFAULT_SNAP_DIR, help="root of the COLIBRE runs")
    p.add_argument("--save", default=DEFAULT_SAVE_DIR, help="root output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Snapshot and output folders
# ---------------------------------------------------------------------------
def open_snapshot(args):
    """Build the snapshot path from the arguments, check it exists, create the output folders.
       Returns (snap, snap_filename, save_path)."""
    snap_filename = os.path.join(args.snap_dir, args.run, "SOAP-HBT",
                                 f"colibre_with_SOAP_membership_{args.snap:04d}.hdf5")
    if not os.path.isfile(snap_filename):
        sys.exit(f"Snapshot not found: {snap_filename}")      # message on stderr, exit code 1
    save_path = os.path.join(args.save, args.run.replace("/", "_"), f"snap_{args.snap:04d}")
    make_dirs(save_path)
    snap = sw.load(snap_filename)
    return snap, snap_filename, save_path


def print_snapshot_header(snap, run, snap_filename, save_path):
    """Basic information on the snapshot (metadata only)."""
    meta = snap.metadata
    print(f"\n----------- Inspecting the Colibre snapshot {run} -----------")
    print(f"     Snapshot file                  :     {snap_filename}")
    print(f"     Figures saved in               :     {save_path}")
    print(f"     Available groups               :     {snap}")
    print(f"     Box size                       :     {meta.boxsize}")
    print(f"     Cosmic Time                    :     {meta.t.to('Myr').value:.3f} Myr")
    print(f"     Redshift (scale factor)        :     {meta.z:.2f} ({meta.a:.2f})")
    print(f"     Hubble parameter H(z)          :     {meta.cosmology.H(meta.z):.2f}")
    print(f"     Number of Gas particles        :     {meta.n_gas}")
    print(f"     Number of DM particles         :     {meta.n_dark_matter}")
    print(f"     Number of stellar particles    :     {meta.n_stars}")
    print(f"     Number of black hole particles :     {meta.n_black_holes}")


def make_dirs(base):
    for d in FIG_DIRS.values():
        os.makedirs(os.path.join(base, d), exist_ok=True)


def fig_path(base, topic, filename):
    """Full path of a figure: base / FIG_DIRS[topic] / filename."""
    return os.path.join(base, FIG_DIRS[topic], filename)


def save_figure(fig, base, topic, filename, dpi=300):
    """Save fig in base / FIG_DIRS[topic] / filename, close it and print the path in the log."""
    path = fig_path(base, topic, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"     [saved] {path}")


# ---------------------------------------------------------------------------
# Gas fields inspection (metadata only)
# ---------------------------------------------------------------------------
def print_gas_fields(meta):
    """Print in alphabetical order the gas fields available in the snapshot
       (swiftsimio name and HDF5 dataset) and, for the fields stored as named
       columns, the names of their sub-fields. Reads metadata only."""

    gas_props  = meta.gas_properties
    named_cols = gas_props.named_columns
    fields     = sorted(zip(gas_props.field_names, gas_props.field_paths))
    w = max(len(n) for n, _ in fields)    # space printing according to the swiftsimio name length

    print(f"\n[+] Available gas fields ({len(fields)}):")
    print(f"     {'#':>3}  {'swiftsimio name':<{w}} {'HDF5 dataset'}")
    print("     " + "-"*95)
    for i, (name, path) in enumerate(fields, start=1):
        print(f"     {i:>3}  {name:<{w}} {path}")
        columns = named_cols.get(path)
        if columns:
            print(f"     {'':>3}    -> columns: {', '.join(columns)}")
