from setuptools import setup, find_packages

setup(
    name="inboxpilot-openenv",
    version="1.0.0",
    description="InboxPilot professional email triage and operations environment",
    author="InboxPilot Team",
    packages=find_packages(include=["app*", "server*"]),
    python_requires=">=3.11",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "openai",
        "httpx",
        "pytest",
        "python-dotenv",
        "pyyaml",
        "fastmcp",
        "openenv-core>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "inboxpilot-server=server.app:main",
        ],
    },
)
