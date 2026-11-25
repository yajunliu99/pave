# Signal and sign policies

JSON overrides of `fusion/control_sampling.DEFAULT_POLICY`, passed with `scripts/pipeline/run_signals.py --policy`.

- `cambridge_pre_construction.json`: the signals and signs of Broadway & Ames Street, Cambridge MA, from before the construction that began around 2022–2023. The worked example in `examples/broadway_ames_2021/` was read with it.
  - `preferred_date_window`: panoramas captured from January 2019 to December 2022.
  - `date_search`: around each camera position, the metadata service is asked at points every 3 m along and across the road, so that panoramas in the window turn up where it would otherwise return a later one.
  - `known_panoramas`: per approach, panoramas found in Google Maps' time travel that the location search does not return; they are judged like any other.
  - `date_fallback`: false, so a position with no panorama in the window is left without a view.
  - `max_metadata_requests`: raised for the date search. Metadata requests are free; images are billed.
