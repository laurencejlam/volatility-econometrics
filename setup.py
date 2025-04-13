from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="volfore",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced volatility forecasting using time series and machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/volfore",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "statsmodels>=0.14.0",
        "arch>=5.4.0",
        "yfinance>=0.2.0",
        "xgboost>=1.7.0",
        "tensorflow>=2.12.0",
        "pandas-datareader>=0.10.0",
        "plotly>=5.14.0",
    ],
) 