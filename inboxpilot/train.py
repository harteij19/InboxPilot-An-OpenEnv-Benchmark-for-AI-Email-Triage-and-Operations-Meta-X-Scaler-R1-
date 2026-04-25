from __future__ import annotations


def train() -> dict[str, str]:
    """Placeholder train entrypoint for hackathon demos.

    No heavy training, checkpoints, or downloads are performed.
    """
    return {
        "status": "ok",
        "message": "Lightweight rule-based policy loaded. No training required.",
    }


if __name__ == "__main__":
    print(train())
