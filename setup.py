from setuptools import setup, find_packages

setup(
    name="inboxpilot-openenv",
    version="1.0.0",
    description="InboxPilot — Professional email triage and operations environment.",
    author="InboxPilot Team",
    packages=find_packages(include=["app*", "server*"]),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.6.0",
        "openai>=1.12.0",
        "httpx>=0.27.0",
        "pytest>=8.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "fastmcp",
        "openenv-core>=0.2.0"
    ],
    entry_points={
        "console_scripts": [
            "server=server.app:main",
        ],
    },
)
