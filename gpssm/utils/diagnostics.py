"""Diagnostics helpers for gpssm."""

def summarize_fit(model):
    return {
        "is_fitted": getattr(model, "is_fitted", False),
        "parameters": {k: v for k, v in getattr(model, "__dict__", {}).items() if k != "is_fitted"},
    }
