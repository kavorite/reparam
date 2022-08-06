from setuptools import find_namespace_packages, setup

setup(
    name="reparam",
    version="1.0",
    description="miscellaneous reparameterization schemes in dm-haiku",
    url="https://github.com/kavorite/reparam",
    install_requires=["dm_haiku>=0.0.6", "jax>=0.2", "numpy>=1.7"],
    packages=find_namespace_packages(),
)
