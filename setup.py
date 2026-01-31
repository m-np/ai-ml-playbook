from setuptools import setup, find_packages

setup(
    name="agentic-ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers",
        "langchain",
        "openai",
        "faiss-cpu",
        "numpy",
    ],
)