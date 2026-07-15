"""
calibration/ — the source of MEASURED TRUTH for turning sensor voltage into ppm.

This package holds DATA and REFERENCE only (node constants, the datasheet curve, the
Mitchell spec, docs). The runtime ENGINE stays in ``physics/sensor_frontend.py``; it
consumes what lives here. Data flows one way:

    calibration/nodes/nodeN.py  ──apply_node──▶  sensor_frontend globals  ──▶  kernel + gate

Nothing here imports the engine's power-law/Mitchell math or re-implements it. See
``calibration/README.md`` for the full pipeline map and the reconciliation rule.
"""
