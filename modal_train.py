import os
import subprocess

import modal

# Instructions for install flash-attn taken from this Modal guide doc:
# https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
n_proc_per_node = 8

base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "libibverbs-dev", "libibverbs1")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "datasets==3.6.0",
        "tiktoken==0.9.0",
        "wandb==0.19.11",
        "tqdm==4.67.1",
    )
)

image = base_image.add_local_dir(
    LOCAL_CODE_DIR,
    remote_path=REMOTE_CODE_DIR,
)

app = modal.App("nanoGPT", image=image)
volume = modal.Volume.from_name("nanogpt", create_if_missing=True)

@app.function(
    gpu=f"H100!:{n_proc_per_node}",
    volumes={
        "/data/fineweb10B": volume,
    },
    timeout=4 * 60,
    image=(
        base_image.pip_install(
            # Modded nanogpt requires a nightly version of torch which is no longer hosted.
            # Use this custom hosted version instead.
            # https://github.com/KellerJordan/modded-nanogpt/issues/91#issuecomment-2831908966
            "https://github.com/YouJiacheng/pytorch-nightly-whl-archive/releases/download/v2.7.0.dev20250208/torch-2.7.0.dev20250208+cu126-cp312-cp312-manylinux_2_28_x86_64.whl",
            extra_index_url="https://download.pytorch.org/whl/nightly/cu126",
        ).add_local_dir(
            LOCAL_CODE_DIR,
            remote_path=REMOTE_CODE_DIR,
        )
    ),
)
def speedrun_modded_single_node():
    print(os.environ["MODAL_TASK_ID"])
    from torch.distributed.run import parse_args, run

    args = [
        "--standalone",
        f"--nproc-per-node={n_proc_per_node}",
        "/root/train_gpt.py",
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))