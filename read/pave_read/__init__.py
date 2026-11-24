"""PAVE Read: lanes, lane use and turning movements at a junction, read by a vision-language model from a satellite
crop and street-level views, and anchored to a GMNS network.

Subpackages:
    harness    arm windows and measurements (geometry), prompts and validators of each step (stages), and the
               street-view sheets and the blind reading (views)
    autoloop   crops, rulers and frames along a link, and the projection of traced lanes into street views
    fusion     satellite and street-view acquisition for a node, and the validation of a blind reading
    hybrid     model requests and their cache (inference), file helpers, geometry helpers and the system prompt

vision_client speaks to any OpenAI-compatible API; vlm_bridge is a file queue through which agents answer instead.
The entry points are scripts/pipeline/acquire_site_imagery.py and scripts/pipeline/run_read.py.
"""
