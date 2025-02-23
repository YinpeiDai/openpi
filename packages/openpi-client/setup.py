from setuptools import setup, find_packages

setup(
    name="openpi_client",  # Replace with the actual package name
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "dm-tree>=0.1.8",
        "msgpack>=1.0.5",
        "numpy>=1.21.6",
        "pillow>=9.0.0",
        "tree>=0.2.4",
        "websockets>=11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust license if needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
