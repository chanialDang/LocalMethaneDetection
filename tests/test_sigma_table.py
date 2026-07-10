"""
test_sigma_table.py — provenance guard for the Briggs σ lookup table.

CLAUDE.md claims briggs_dispersion_sigma.csv is byte-identical to what
generate_sigma_table.py produces (reproducible provenance). This locks that claim:
regenerate the CSV in memory and compare it to the committed file exactly, so a hand
edit to the CSV — or a drift in the generator — can never slip in unnoticed.
"""
import csv
import io
from pathlib import Path

from physics import generate_sigma_table as gen


def test_committed_sigma_table_matches_generator_byte_for_byte():
    rows = gen.build_rows()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))   # same schema as write_csv
    writer.writeheader()
    writer.writerows(rows)
    generated = buf.getvalue()                               # csv default → \r\n endings

    csv_path = Path(gen.__file__).parent / "briggs_dispersion_sigma.csv"
    committed = open(csv_path, newline="").read()            # preserve \r\n for a true compare

    assert generated == committed, (
        "briggs_dispersion_sigma.csv is NOT byte-identical to generate_sigma_table.py — "
        "regenerate it with `python3 physics/generate_sigma_table.py` or revert the edit."
    )


def test_generator_self_check_anchors_hold():
    # The generator's own hand-checked Briggs anchors (independent of the CSV).
    gen._self_check()                                        # raises AssertionError on drift
