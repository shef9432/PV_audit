from setuptools import setup, find_packages

setup(
    name="pv-audit",
    version="0.1.2",
    packages=find_packages(),
    install_requires=["numpy", "opencv-python", "pandas", "seaborn", "matplotlib"],
    author="shef9432",
    description="Industrial AI Robustness Audit SDK"
)