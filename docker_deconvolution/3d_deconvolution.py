import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
import time
import pyclesperanto_prototype as cle
from clij2fft.richardson_lucy_dask import richardson_lucy_dask
import gc
import os
import glob
from pathlib import Path

def main():
    # -----------------------
    # Parse CLI arguments
    # -----------------------
    parser = argparse.ArgumentParser(description="Single-image 3D deconvolution using CLIJ2-FFT")
    parser.add_argument("--input", required=True, type=str, help="Path to input TIFF image")
    parser.add_argument("--output", required=True, type=str, help="Path to save deconvolved image")
    parser.add_argument("--psf", required=True, type=str, help="Path to PSF TIFF image")
    parser.add_argument("--iterations", default=20, type=int, help="Number of RL iterations")
    parser.add_argument("--reg", default=0.0002, type=float, help="Regularization factor")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    psf_path = Path(args.psf)

    # -----------------------
    # Select GPU
    # -----------------------
    print("Selecting GPU...")
    cle.select_device("GPU")
    device = cle.get_device()
    print("Using device:", device.name)

    # -----------------------
    # Load PSF and input image
    # -----------------------
    print(f"Loading PSF: {psf_path}")
    psf = tiff.imread(psf_path).astype(np.float32)

    print(f"Loading image: {input_path}")
    img = tiff.imread(input_path).astype(np.float32)
    print(f"Image shape: {img.shape}, size: {img.nbytes / 1e9:.2f} GB")

    # -----------------------
    # Deconvolution
    # -----------------------
    start_time = time.time()
    print("Starting Richardson–Lucy deconvolution...")

    decon = richardson_lucy_dask(
        img,
        psf,
        args.iterations,
        args.reg,
        num_devices=1  # must be 1 for single GPU
    )

    elapsed = time.time() - start_time
    print(f"Deconvolution finished in {elapsed/60:.1f} minutes")

    # -----------------------
    # Save result
    # -----------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(output_path, decon.astype(np.float32))
    print(f"Saved deconvolved image to {output_path}")

    # -----------------------
    # Free GPU memory
    # -----------------------
    del img, decon
    gc.collect()
    cle.clear()

    print("Done.")

if __name__ == "__main__":
    main()
