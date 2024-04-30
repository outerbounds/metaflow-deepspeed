from setuptools import setup, find_namespace_packages

version = "0.0.9"

setup(
    name="metaflow-deepspeed",
    version=version,
    description="An EXPERIMENTAL Deepspeed decorator for Metaflow",
    author="Eddie Mattia",
    author_email="eddie@outerbounds.com",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)