import json
import logging
import argparse

from cerebras.sdk.client import SdkCompiler
from cerebras.appliance import logger

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Compile a Cerebras stencil layout with configurable parameters.")
parser.add_argument("--kernel-dim-x", type=int, default=16, help="Kernel dimension in X")
parser.add_argument("--kernel-dim-y", type=int, default=16, help="Kernel dimension in Y")
parser.add_argument("--inp-rows", type=int, default=1024, help="Number of input rows")
parser.add_argument("--inp-cols", type=int, default=1024, help="Number of input columns")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")

args = parser.parse_args()

# WSE3 cores
fabric_dim_x = 762 
fabric_dim_y = 1172


# Instantiate compiler using a context manager
# Disable version check to ignore appliance client and server version differences.
with SdkCompiler(disable_version_check=True) as compiler:

    # Launch compile job
    artifact_path = compiler.compile(
        ".",
        "layout.csl",
        f'--fabric-dims={fabric_dim_x},{fabric_dim_y} --fabric-offsets=4,1 --params=kernel_dim_x:{args.kernel_dim_x},kernel_dim_y:{args.kernel_dim_y},M:{args.inp_rows},N:{args.inp_cols},iterations:{args.iterations} -o out --memcpy --channels=1 --arch=wse3',
        "."
    )

# Write the artifact_path to a JSON file
with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path,}, f)