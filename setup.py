from setuptools import setup, find_packages

setup(
    name="classification-mlops",
    version="0.1.0",
    author="Anurag Trivedi",
    author_email="aanuargtrivedi007@gmail.com,
    description="An MLOps pipeline for classification tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/QsingularityAi/MLOps_Classification.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.6.0",
        "mlflow>=1.20.0",
        "pytest>=6.2.5",
    ],
    entry_points={
        "console_scripts": [
            "train=classification_mlops.train:main",
            "predict=classification_mlops.predict:main",
        ],
    },
)