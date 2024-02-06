from setuptools import setup

setup(
    name="edge-transformer",
    py_modules=["edge_transformer"],
    install_requires=[
        "torch==2.1.0",
        "torch_geometric==2.4.0",
        "hydra-core==1.3.2",
        "wandb==0.15.12",
        "loguru==0.7.2",
        "tqdm==4.66.1",
        "brec==1.0.0",
    ],
    version="0.0.1",
)
