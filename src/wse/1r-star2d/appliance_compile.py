import json
import logging

from cerebras.sdk.client import SdkCompiler
from cerebras.appliance import logger

logging.basicConfig(level=logging.INFO)


kernel_dim_x = 750
kernel_dim_y = 750
fabric_dim_x = 762  # kernel_dim_x + 7
fabric_dim_y = 1172 # kernel_dim_y + 2

inp_rows = 48000
inp_cols = 48000
iterations = 100

# Instantiate compiler using a context manager
# Disable version check to ignore appliance client and server version differences.
with SdkCompiler(disable_version_check=True) as compiler:

    # Launch compile job
    artifact_path = compiler.compile(
        ".",
        "layout.csl",
        f'--fabric-dims={fabric_dim_x},{fabric_dim_y} --fabric-offsets=4,1 --params=kernel_dim_x:{kernel_dim_x},kernel_dim_y:{kernel_dim_y},M:{inp_rows},N:{inp_cols},iterations:{iterations} -o out --memcpy --channels=1 --arch=wse3',
        "."
    )

# Write the artifact_path to a JSON file
with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path,}, f)