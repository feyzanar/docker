from cellpose import core, utils, io, models, denoise, metrics, plot
from cellpose.io import imread
import torch
import tifffile as tiff
import os, shutil
import glob
import numpy as np
from pathlib import Path
import argparse
import os

# Set umask so new files are group-writable
os.umask(0o0002)

def main():
    parser = argparse.ArgumentParser(
        description="Cellpose argparse"
    )
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("output", help="Path to the input file")
    parser.add_argument("model", help="Path to the output file")
    parser.add_argument("--diameter", "-d", default = 15, help="diameter for the model", type=float)
    parser.add_argument("--anisotropy", "-a", default = 1.5, help="anisotropy", type=float)
    parser.add_argument("--minsize", "-m", default = -1., help="minimun size", type=float)
    parser.add_argument("--channels", "-c", default = [0,0], help="channels for the model", type=int, nargs='+')
    parser.add_argument("--smooth", "-s", default = 0, help="smoothing parameter for the model (removes ring artifacts)", type=int)
    parser.add_argument("--image", "-i", default = "", help="input name of the image to match between NAS and Cluster", type=str)
    parser.add_argument("--denoise", "-n", default = "denoise_cyto3",
                        choices=["denoise_cyto3", "denoise_nuclei"],
                        help="denoise model type: 'denoise_cyto3' (default) or 'denoise_nuclei'", type=str)
    parser.add_argument("--verbose", "-v",action="store_true",help="Increase output verbosity")
    
    args = parser.parse_args()
    print(args.channels)
    
    if args.verbose:
        print("Verbose mode enabled")
    
    print("Processing file  : ", args.input)
    print("Processing model : ", args.model)
    print("Denoise model    : ", args.denoise)

    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')
    print("CUDA device ", torch.cuda.torch.cuda.get_device_properties())
    
    # Setup logging for Cellpose
    io.logger_setup()
    
    ### MODEL
    model_path = str(args.model)
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    ### PARAMETERS
    # Define channels
    channels = [args.channels]  # cytoplasm: 1, nucleus: 2
    
    # Segmentation parameters
    diameter = args.diameter  # in pixels
    cellprob_threshold = -1
    
    # Input folder containing images
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist
    
    # Process all files in the folder
    for img_file in input_folder.glob("*.tif"):
        if args.image!="" and args.image!=img_file.name:continue
        # Read image
        img = imread(img_file)
        print(f'Loaded image {img_file.name} with shape: {img.shape} and data type {img.dtype}')
    
        # Generate output file name
        output_file_base = output_folder / img_file.stem  # Use the base name without extension
        print("========================output_file_base ",output_file_base)
        
        # USE cellposeDenoiseModel FOR DENOISING THE IMAGE
        from cellpose import denoise
        dn = denoise.DenoiseModel(model_type=args.denoise,
                                  gpu=True,
                                  chan2=False)
        img = dn.eval([img], channels=channels, diameter=diameter)
    
        # Perform segmentation
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            cellprob_threshold=cellprob_threshold,
            do_3D=True,
            anisotropy=args.anisotropy,
            min_size=args.minsize,
            dP_smooth=args.smooth
        )


        print("name just before save (img_file.stem) ",img_file.stem)
        print("type img   ",type(img)  ,"  len(img)   ", len(img))
        print("type masks ",type(masks),"  len(masks) ", len(masks))
        print("type flows ",type(flows),"  len(flows) ", len(flows))
        
        # Save output masks to tiffs/pngs or txt files for ImageJ
        io.save_masks(
            img,
            masks,
            flows,
            #str(output_file_base),
            [img_file.stem],
            channels=channels,
            png=False,  # Save masks as PNGs
            tif=True,  # Save masks as TIFFs
            save_txt=True,  # Save txt outlines for ImageJ
            save_flows=False,  # Save flows as TIFFs
            save_outlines=False,  # Save outlines as TIFFs
            save_mpl=False,  # Make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
            savedir=output_folder
        )

        # Set permissions: read+write for owner, group, and others (no execute)
        os.chmod(args.output, 0o0777)
        output_tif = str(output_file_base) + '_cp_masks.tif'
        if os.path.exists(output_tif):
            os.chmod(output_tif, 0o0666)


if __name__ == "__main__":
    main()
