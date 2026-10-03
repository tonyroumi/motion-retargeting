"""
Basic smoke test for loading CMU AMASS data through AMASSAdapter.
"""

from motion_retargeting.data.adapters import AMASSAdapter, CMUAdapter


if __name__ == "__main__":
    AMASSAdapter(device="cuda").load()

    CMUAdapter(device="cuda").load()
