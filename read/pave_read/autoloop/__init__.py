"""Draft -> check -> review -> revise loops for candidate lane geometry.

The hybrid pipeline consumes candidate regions that were traced once, outside
the code. This package produces them inside the pipeline and lets a bounded
review loop correct them:

    frames          section crop windows from road-network geometry
    strips          lane-strip geometry and the edits a review may apply
    checks          coordinate-only heuristics that raise review issues
    controller      the bounded loop: stop reasons, oscillation, budgets
    render          travel-up crops, rulers, boundary overlays, region panels
    geometry_stage  prompts, validators and the per-site runner
    evaluate        post-run comparison with reference polygons

Lane-count priors from the network only size the crop window. They are never
sent to the model, and reference geometry is read only by `evaluate`.
"""
