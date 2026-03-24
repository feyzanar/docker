#python sendjobs.py --name cellpose-test-2 --image registry.rcp.epfl.ch/upoates-helsens/cellpose-env:v0.1 --gpu 1. --input /scratch/feyza/input/test2c --output /scratch/data/output/feyza --model /scratch/data/input/feyza/cellpose_training/2d/models/CP_20241007_h2bxncad -d 30 -c 2 1
import argparse
import os, sys, re
import subprocess
import glob
from pathlib import Path

#__________________________________________________________
def getCommandOutput(command):
    p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE,universal_newlines=True)
    (stdout,stderr) = p.communicate()
    return {"stdout":stdout, "stderr":stderr, "returncode":p.returncode}


#__________________________________________________________
def main():
    parser = argparse.ArgumentParser(
        description="Cellpose send jobs"
    )

    parser.add_argument("--name",   help="Name of the job")
    parser.add_argument("--image",  help="Name of the image")
    parser.add_argument("--gpu",  help="Number of GPUs", default=1, type=float)
    parser.add_argument("--nas",    help="Path to the input files on the NAS", type=str)

    parser.add_argument("--input",  help="Path to the input file (on the cluster)")
    parser.add_argument("--output", help="Path to the output directory (on the cluster)")
    parser.add_argument("--model",  help="Path to the model file (on the cluster)")
    parser.add_argument("--diameter", "-d", default = 15, help="diameter for the model", type=float)
    parser.add_argument("--anisotropy", "-a", default = 1.5, help="anisotropy", type=float)
    parser.add_argument("--minsize", "-m", default = -1, help="minimum size", type=float)
    parser.add_argument("--channels", "-c", default = [0,0], help="channels for the model", type=int, nargs='+')
    parser.add_argument("--smooth", "-s", default = 0, help="smoothing parameter for the model (removes ring artifacts)", type=int)
    parser.add_argument("--denoise", "-n", default = "denoise_cyto3",
                        choices=["denoise_cyto3", "denoise_nuclei"],
                        help="denoise model type: 'denoise_cyto3' (default) or 'denoise_nuclei'", type=str)
    parser.add_argument("--test",  "-t", action="store_true",help="Run the first image as a test")
    parser.add_argument("--verbose",  "-v", action="store_true",help="Increase output verbosity")
    parser.add_argument("--dry-run",  action="store_true",help="do not run the command, just print it")
    
    args = parser.parse_args()
    channels=" ".join(str(x) for x in args.channels)

    nas_images = os.path.join(args.nas, "*.tif")
    if not os.path.exists(args.nas):
        print('Path to the input files on the NAS ===>{}<=== does not exist. Maybe UPOATES nas not mounted?'.format(args.nas))
        sys.exit(3)

    count=0
    for img_file in glob.glob(nas_images):
        if count>0 and args.test:break

        img=img_file.split('/')[-1]
        img_base = img.split('.')[0]
        safe_img_name = re.sub(r'[^a-z0-9]+', '-', img_base.lower()).strip('-')
        name = f"{args.name}-{safe_img_name}"
        print('sending image = ',img, '  with job name  ', name)
        cmd='runai submit --name {} --image {} --gpu {} --existing-pvc claimname=upoates-scratch,path=/scratch --command -- /usr/bin/python3 /opt/scripts/3d_cellpose.py {} {} {} --diameter {} --channels {} --image "{}" --anisotropy {} --minsize {} --smooth {} --denoise {}'.format(
            name, args.image, args.gpu, args.input, args.output, args.model,
            args.diameter, channels, img, args.anisotropy, args.minsize, args.smooth, args.denoise)

        print(cmd)
        count+=1
        if args.dry_run==False:
            outputCMD = getCommandOutput(cmd)
            stderr=outputCMD["stderr"].split('\n')
            stdout=outputCMD["stdout"].split('\n')

            print('error:',stderr)
            print('output:',stdout)

'''
e.g.
python sendjobs.py --name cp-downs-ctrl \
  --image registry.rcp.epfl.ch/upoates-fnarslan/cellpose:v0.2 \
  --gpu 1. \
  --nas /Volumes/upoates/common/ForRCP/input/fnarslan/reg_cropped_ctrl \
  --input /scratch/data/input/fnarslan/reg_cropped_ctrl \
  --output /scratch/data/output/fnarslan/reg_cropped_ctrl_cp \
  --model /scratch/data/input/fnarslan/cellpose_training/2d/models/nuclei_h2b \
  -d 6 -c 1 0 -a 1 \
  -n denoise_nuclei
'''

main()
