"""net2cell VLM-augmented audit and correction layer.

Companion repository for the paper:
    "From Topology to Legality: Accurate Multi-Resolution Traffic Simulation
     Networks through Vision-Language Reasoning"

The package exposes a movement-first correction primitive:
    * ``reconciliation_movements.run_movement_first_reconciliation``
      consumes a satellite VLM audit and a GSV per-lane VLM audit and
      produces a ``CorrectionSpec``;
    * ``apply_corrections.apply_corrections_to_csvs`` writes the spec
      onto the GMNS CSVs;
    * ``apply_corrections.rebuild_with_net2cell`` regenerates the
      mesoscale and microscale layers via net2cell;
    * ``apply_corrections.diff_multi_resolution`` produces the
      per-layer diff against the auto-gen baseline.

Submodules (top-down by responsibility):
    satellite_client          - Mapbox Static Images API tile fetcher
    gsv_tile_client           - Google Street View pano tile fetcher +
                                1280x1280 fov=60 perspective projector
    vlm_client                - CreateAI VLM client (Claude / GPT-5 /
                                Gemini / Qwen) + per-lane audit prompt
    satellite_vlm_audit       - 4-pass satellite audit prompt + parser
    reconciliation_movements  - movement-first reconciliation engine
                                (cluster + transition nodes)
    reconciliation            - Class III sign-restriction reconciliation
    apply_corrections         - CorrectionSpec dataclasses + CSV apply +
                                net2cell rebuild + multi-res diff
    topology_checks           - post-apply topology lint
    gmns_io                   - GMNS schema-aware CSV I/O helpers
    data_models               - shared dataclasses (LatLon, TurnType,
                                Approach, MovementCandidate)

The legacy v0.2 "evidence-fused" stack (cli, pipeline, fusion,
evidence_*, gsv_client (Static API), projection, error_injection,
verdicts_to_spec) was retired during cleanup; this package now
exclusively implements the v0.3 movement-first architecture.
"""

__version__ = "0.3.0"
