# Acquisition policies

JSON overrides of `fusion/lane_acquisition.POLICY`, passed with `scripts/pipeline/acquire_site_imagery.py --policy`.

- `cambridge_pre_construction.json`: imagery of Broadway & Ames Street, Cambridge MA, from before the construction that began around 2022–2023. The worked example in `examples/broadway_ames_2021/` was read from it.
  - Satellite: MassGIS 2021 orthos (spring 2021, leaf-off, 15 cm), assembled over the usual Mapbox viewport.
  - Street views: Google's own panoramas captured before January 2023, found by searching around each requested position.
    Where there is none, the earliest usable one found there is used instead and marked `date_fallback`.
  - `pinned_panoramas`: the November 2020 panoramas for the positions where the metadata service returns only later captures.
    Google Maps' time travel lists them; they were picked there, one per position, each on the carriageway of its section.
  - `same_carriageway`: Broadway is a divided road here, so a panorama on the other side of the median is not used for a section.
  - Use it with the 2021 Kendall Square network in `examples/broadway_ames_2021/network/`.
