#!/usr/bin/env python3
"""Compatibility entry point for the independent upstream RMVPE reference.

The old locally duplicated network/frontend silently used Slaney frequency
spacing. It is retired: generate rmvpe_htk_* and e2e_htk_* assets instead, retaining
historical rmvpe_ref_* / vits_ref_* only as fixed-input numerical controls.
"""
from gen_rmvpe_htk_reference import main

if __name__ == "__main__":
    main()
