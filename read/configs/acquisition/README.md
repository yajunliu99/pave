# Acquisition policies

JSON overrides of `fusion/lane_acquisition.POLICY`, passed with `run_site_pipeline.py --acquisition-policy`.

- `cambridge_pre_construction.json`: imagery of Broadway & Ames Street, Cambridge MA, from before the construction that began around 2022–2023.
  - Satellite: MassGIS 2021 orthos (spring 2021, leaf-off, 15 cm), assembled over the usual Mapbox viewport.
  - Street views: Google's own panoramas captured before January 2023, found by searching around each requested position.
    Where there is none (the west arm of Broadway, all of Ames Street), the earliest usable one found there is used instead and marked `date_fallback`.
  - `pinned_panoramas`: the November 2020 panoramas for the positions where the metadata service returns only later captures.
    Google Maps' time travel lists them; they were picked there on 2026-10-11, one per position, each on the carriageway of its section.
  - `same_carriageway`: Broadway is a divided road here, so a panorama on the other side of the median is not used for a section.
- `configs/control_policies/cambridge_pre_construction.json` (signal audit) lists panoramas per approach in `known_panoramas`: November 2020 at the stop line, in the junction and 25–50 m upstream, and two in the middle of the junction from October 2020 and October 2019. The audit's date search adds them to the panoramas it finds near each requested position.
  - Use it with the 2021 Kendall Square network (`data/cambridge_kendall_2021/`; in vlm4net, `examples/broadway_ames_2021/network/`) and with `configs/control_policies/cambridge_pre_construction.json` for the signal audit (`--control-policy`).
