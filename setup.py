from setuptools import setup, find_packages

setup(
    name="classify",
    version="0.1.0",
    author="Herve Yav",
    author_email="rvyav@hotmail.com",
    description="Multiclass classification of new articles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/iamherve/classify",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Operating System :: OS Independent",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
    ],
    python_requires=">=3.6",
)
