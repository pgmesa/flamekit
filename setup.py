import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r", encoding='utf-8') as fh:
    req = fh.readlines()
    requirements = [line.strip().replace("\n", "") for line in req]

setuptools.setup(
    name='flamekit',
    version='0.2.0',
    author="Pablo García Mesa",
    author_email="pgmesa.sm@gmail.com",
    description="Minimalistic toolkit for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgmesa/flamekit",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )