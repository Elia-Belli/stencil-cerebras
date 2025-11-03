import json
from cerebras.sdk.client import SdkCompiler
import logging
from cerebras.appliance import logger
logging.basicConfig(level=logging.INFO)


fabric_dim_x = 16
fabric_dim_y = 16
kernel_dim_x = fabric_dim_x + 7
kernel_dim_y = fabric_dim_y + 7
inp_rows = 1024
inp_cols = 1024
iterations = 1000

# Instantiate compiler using a context manager
# Disable version check to ignore appliance client and server version differences.
with SdkCompiler(disable_version_check=True) as compiler:

    # Launch compile job
    artifact_path = compiler.compile(
        ".",
        "layout.csl",
        f'--fabric-dims={fabric_dim_x},{fabric_dim_y} \
        --fabric-offsets=4,1 \
        --params=kernel_dim_x:{kernel_dim_x},kernel_dim_y:{kernel_dim_y},M:{inp_rows},N:{inp_cols},iterations:{iterations} \
        -o out --memcpy --channels=1',
        "."
    )

# Write the artifact_path to a JSON file
with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path,}, f)