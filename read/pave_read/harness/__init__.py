"""Lean Read pipeline: per road arm, one trace of both carriageways, one independent street-view reading per approach,
a re-trace only where the two disagree, and one lane-use and movement reading per approach.

No reference data (UTDF, OSM lane counts, human labels) enters a prompt or decides which request is made."""
