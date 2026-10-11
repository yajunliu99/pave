"""VLM client for ASU CreateAI vision endpoint + per-approach audit orchestration.

Three layers:

    CreateAIClient.query_vision     low-level: one image + one query → JSON response
    CreateAIClient.audit_approach   mid-level: stitch pavement+signs, build prompt,
                                    parse verdicts for one sample
    run_cluster_audit               high-level: walk a GSV sample manifest, produce
                                    Verdicts for every approach of the cluster

The mid-level verdicts are intentionally structured and stable-keyed
(mvmt_txt_id, direction) so they can be fed into CorrectionSpec without
manual glue.

CLI smoke test:

    python -m movement_fixer.vlm_client \\
        --gsv-dir    ./cache/gsv_mill_riosalado \\
        --movement-csv "data/Tempe Rio Salado Pkwy/output/movement.csv" \\
        --cluster-node 363 \\
        --out-json   ./cache/vlm_audit_mill.json
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import requests

from .vlm_config import (
    DEFAULT_MODEL_NAME, MODEL_PRESETS,
    REQUEST_PROTOCOL_VERSION, REQUEST_SOURCE, ModelRoutingError,
    resolve_model, validate_model_identity,
)


log = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_S = 90.0
_DEFAULT_MAX_RETRIES = 3
_SYSTEM_PROMPT = (
    "You are a transportation engineer auditing lane usage at a signalized "
    "intersection based on street-view imagery. You classify each lane's "
    "function from painted pavement arrows, lane geometry, and any posted "
    "signs. You are allowed and encouraged to REASON beyond pure observation "
    "using standard US traffic engineering principles — but you must flag each "
    "classification as either 'observed' (direct visual evidence) or 'inferred' "
    "(applied engineering knowledge to fill a gap), with a confidence level. "
    "You output ONLY valid JSON matching the requested schema."
)

_VERIFICATION_PROMPT_TEMPLATE = """\
Context:
  Intersection: {intersection_name}
  Approach direction: {approach_direction} (driver faces this direction approaching the stop bar)
  Declared lane count (per OSM / auto-gen): {num_lanes} travel lanes.
  Sample location: {sample_location_m:.0f}m upstream of the stop bar.

  IMPORTANT — the declared lane count may be wrong. Count what you actually
  see in the image. If you see MORE than {num_lanes} motor-vehicle travel
  lanes at the stop bar, enumerate ALL of them in `per_lane` (e.g., lanes
  1..{num_lanes}+1) and set `observed_motor_vehicle_lane_count` accordingly.
  If you see FEWER (e.g., because a lane drops or merges before the stop
  bar), report only what you see. Always number lanes 1..N from LEFT to
  RIGHT as the driver sees them, where N = what you actually observe.

Image:
  Two views stitched top-to-bottom:
    TOP half:    pitch angled DOWN — shows pavement arrows and lane markings
    BOTTOM half: pitch angled UP   — shows overhead signs and ground-mounted sign poles

  Pavement arrows are commonly painted within the last 10-20m before the stop bar
  — in the UPPER third of the pavement half of the image, NOT the foreground
  directly under the camera. Scan the full length of each lane all the way to
  the stop bar before concluding no arrow is present.

Task: CLASSIFY EACH LANE'S USAGE. Direct observation is preferred, but you may
apply engineering reasoning to fill gaps — just flag each classification
explicitly as observed vs inferred and provide a brief rationale.

Vocabulary (tags for the `usages` list — may combine when genuinely shared):

  through                - through-directional arrow painted on the lane, OR
                           no turn-only arrow and the lane visibly continues past
                           the stop bar without geometric narrowing.
  left_turn_bay          - a left-directional arrow on the lane, OR the lane
                           visibly ends at the stop bar on the left side of
                           the roadway (widening terminates).
  right_turn_bay         - analogous for right.
  shared_through_left    - REQUIRES one of:
                             (a) a single combination arrow glyph (e.g. ↖↑)
                                 painted on the lane, OR
                             (b) TWO physically separate arrows (a left arrow
                                 AND a through arrow) painted on the SAME
                                 lane, both visible in the image.
                           DO NOT tag shared_through_left when:
                             * Only a left arrow is painted (this is
                               left_turn_bay).
                             * Only a through arrow is painted (this is
                               through).
                             * The lane is leftmost but you only INFER it
                               might be shared because no dedicated left bay
                               is visible. In that case, set based_on=
                               "inferred", confidence at most "medium", and
                               prefer left_turn_bay when the lane visibly
                               ends at the stop bar.
  shared_through_right   - Symmetric requirements (right + through arrows on
                           the same lane, or a combo glyph). DO NOT tag a
                           curb lane as shared_through_right just because it
                           could conceivably host a right turn; require
                           visual evidence of both arrows or a combo glyph.
  u_turn                 - dedicated U-turn bay. Requires an explicit U-turn
                           arrow or a visible "U-Turn Permitted" sign.
  bus_transit            - bus-only / BRT / dedicated transit lane.
  parking_loading        - parking stripes, loading zone, metered parking.
  lane_drops             - lane narrows/ends before the stop bar.
  unknown                - complete occlusion or fully illegible view.

(Note: `bike_lane` is NOT in this vocabulary. A bike lane is not counted as
one of the N travel lanes; report its presence in `bike_lane_present` below.)

Bike-lane disambiguation:
  A bike lane is a NARROW strip (~1.2-1.5 m wide) with one or more of:
    - Green paint covering the lane surface,
    - A bicycle symbol painted on the surface, or
    - A bike-lane sign on a pole next to the curb.
  A motor-vehicle curb lane is WIDER (~3.0-3.7 m) and does not have these
  markings; it may have a right-turn arrow or be unmarked.

  Common failure mode to avoid: when green bike paint extends slightly
  into the adjacent travel lane (e.g., at a conflict-zone marking near
  the stop bar), the curb travel lane is STILL a motor-vehicle lane, not
  a bike lane. Do not subtract the curb lane in that case.

  Counting rule: include the curb lane in `per_lane` if it is at least
  about 3.0 m wide and is NOT marked with a bike symbol or with full
  green paint along its length. Width relative to the bike strip (if
  any) is the most reliable disambiguator.

Reasoning is explicitly welcomed. You may cite:
  - Standard US arterial layout: leftmost lane typically left/shared;
    rightmost typically right/shared; middle lanes usually through.
  - Cross-lane consistency: if lanes 1 and N are dedicated bays, the
    remaining middle lanes are almost certainly through.
  - Geometry: a lane that visibly widens into or narrows out of the stop bar
    is almost certainly a turn bay; lanes of uniform width that continue
    past the stop bar are almost certainly through.

Every classification MUST include:
  - `usages`: one or more tags from the vocabulary.
  - `confidence`: "high" | "medium" | "low".
  - `based_on`: "observed" | "inferred" | "mixed".
  - `rationale`: one sentence — what you saw and any reasoning applied.

Absence of arrow rules:
  - If no arrow AND the lane clearly continues past the stop bar AND no
    sign forbids through: you may classify as `through` (observed or
    inferred depending on how clear the continuation is).
  - If no arrow AND the lane widens then ends at the stop bar: you may
    infer left_turn_bay or right_turn_bay based on which side it widens on.
  - If occluded by a vehicle for >50% of the lane's stop-bar-proximal
    surface: still provide your best guess with confidence="low" and
    based_on="inferred". "unknown" is reserved for genuinely illegible views.

Side channels (report separately, NOT per-lane):
  - `bike_lane_present`: true/false — is a green-painted / bicycle-symbol
    lane adjacent to the N motor vehicle lanes?
  - `restrictions_visible`: list of posted signs readable (e.g. "No U-Turn",
    "No Left Turn", "Right Turn Only", "No Right On Red").

Respond with ONLY valid JSON (no Markdown fences, no prose):
{{
  "approach_direction":                 "{approach_direction}",
  "declared_lane_count":                {num_lanes},
  "observed_motor_vehicle_lane_count":  <int — what you actually counted, may differ from declared>,
  "declared_lane_count_mismatch":       "higher|lower|match",
      // "higher" → declared > observed (auto-gen over-counts; e.g., parking
      //           stripe or shoulder was tagged as a lane),
      // "lower"  → declared < observed (auto-gen under-counts; e.g., a turn
      //           bay widens the stop-bar cross-section and auto-gen missed it),
      // "match"  → equal.
      // BOTH "higher" and "lower" are Class I candidates — flag truthfully.
  "lane_count_mismatch_rationale":      "<one sentence — what you saw that differs>",
  "per_lane": [
    {{
      "lane":        <int, 1..observed_motor_vehicle_lane_count>,
      "usages":      [<tag>, ...],
      "confidence":  "high|medium|low",
      "based_on":    "observed|inferred|mixed",
      "rationale":   "<one sentence>"
    }}
  ],
  "effective_through_lane_count":       <int>,
  "bike_lane_present":                  <bool>,
  "restrictions_visible":               [<sign text or category>, ...],
  "image_quality":                      "good|partially_occluded|unclear",
  "overall_confidence":                 "high|medium|low"
}}
"""


# Signs-only prompt for signal-facing tile samples (10m downstream of the
# stop bar with reverse heading). These samples are reserved for Class~III
# regulatory-sign reading; they do not inform per-lane classification
# (the image is oriented BACKWARD relative to the driver, so the "lanes"
# in view are the far-side exit, not the approach). A narrower prompt
# reduces output-token cost and avoids the per-lane schema overhead.
_SIGNS_ONLY_PROMPT_TEMPLATE = """\
Context:
  Intersection: {intersection_name}
  Approach direction: {approach_direction}
  View: single 1280x1280 driver-view photograph taken roughly 10m
        downstream of the stop bar with REVERSE heading — i.e., looking
        BACK at the mast arm of the {approach_direction} approach from
        the intersection exit side. The mast arm faces traffic approaching
        from {approach_direction}, so signs serving that direction are
        visible in this frame from the front.

Task: READ AND CLASSIFY each posted sign visible in this frame. We care
about signs that affect the lane-and-movement model at the
{approach_direction} stop bar.

Classify each readable sign into ONE of four kinds:

  movement_restriction
      The sign forbids a specific turn type at the cluster level. The
      reconciler will REMOVE the matching movement.
      Examples:
        "No U-Turn"          -> turn_type=uturn
        "No Left Turn"       -> turn_type=left
        "No Right Turn"      -> turn_type=right
      Output: restriction_kind="movement",
              turn_type in {{"uturn","left","right"}},
              lane_position=null.

  lane_use_panel
      The sign assigns a turn type to a SPECIFIC lane rather than
      restricting the whole approach. The reconciler will narrow the
      corresponding movement's lane range.
      Examples:
        "Right Turn Only"    -> turn_type=right,    lane_position=rightmost
        "Left Only"          -> turn_type=left,     lane_position=leftmost
        "Through Only"       -> turn_type=through,  lane_position=null
      Output: restriction_kind="lane_use",
              turn_type as above,
              lane_position in {{"leftmost","rightmost","curb",null}}.

  phase_restriction
      The sign restricts WHEN the movement may occur (which signal phase),
      not whether it exists. The reconciler keeps the movement and
      annotates its phase semantics.
      Examples:
        "No Right On Red"               -> turn_type=right, phase=red_only
        "No Turn On Red"                -> turn_type=any,   phase=red_only
        "Left On Green Arrow Only"      -> turn_type=left,  phase=protected_only
        "Left Turn Yield On Green"      -> turn_type=left,  phase=permissive
      Output: restriction_kind="phase",
              turn_type in {{"uturn","left","right","any"}},
              phase in {{"red_only","protected_only","permissive"}}.

  advisory
      Non-regulatory or generic advisory sign. The reconciler ignores it.
      Examples: "Keep Right", "School Zone", "Bus Stop", "Pedestrian
      Crossing", street-name panels, business-direction signs.
      Output: restriction_kind="advisory",
              turn_type=null, lane_position=null, phase=null.

For each sign whose TEXT you can read (not just see a sign shape),
emit one entry in `signs_classified` with the exact raw text plus the
classification fields. If you can see a sign but cannot read enough text
to classify it, set restriction_kind="advisory" with confidence="low" and
note in rationale that the text was not readable.

Also keep `restrictions_visible` as a flat list of the raw text of every
sign whose text you read (for backward compatibility with downstream
text-only readers).

Respond with ONLY valid JSON (no Markdown fences, no prose):
{{
  "approach_direction":       "{approach_direction}",
  "restrictions_visible":     [<raw sign text>, ...],
  "signs_classified": [
    {{
      "raw_text":          "<exact text as you read it>",
      "restriction_kind":  "movement|lane_use|phase|advisory",
      "turn_type":         "uturn|left|right|through|any|null",
      "lane_position":     "leftmost|rightmost|curb|null",
      "phase":             "red_only|protected_only|permissive|null",
      "confidence":        "high|medium|low",
      "rationale":         "<one short phrase>"
    }}
  ],
  "image_quality":            "good|partially_occluded|unclear",
  "overall_confidence":       "high|medium|low",
  "rationale":                "<one short sentence on what you saw>"
}}
"""


# Single-view (tile-based, pitch=0 horizontal) variant: same schema + vocabulary
# as _VERIFICATION_PROMPT_TEMPLATE, but with the image-description block
# rewritten for a single 1280x1280 driver's-eye photograph rather than a
# stitched pavement+signs composite. Used by
# CreateAIClient.audit_approach_single_image.
_SINGLE_VIEW_PROMPT_TEMPLATE = _VERIFICATION_PROMPT_TEMPLATE.replace(
    """Image:
  Two views stitched top-to-bottom:
    TOP half:    pitch angled DOWN — shows pavement arrows and lane markings
    BOTTOM half: pitch angled UP   — shows overhead signs and ground-mounted sign poles

  Pavement arrows are commonly painted within the last 10-20m before the stop bar
  — in the UPPER third of the pavement half of the image, NOT the foreground
  directly under the camera. Scan the full length of each lane all the way to
  the stop bar before concluding no arrow is present.""",
    """Image:
  Single 1280x1280 horizontal driver-view photograph (pitch=0, fov=60 deg),
  projected from a high-resolution Street View pano tile-set with per-pano
  yaw calibration to compass north. The lower half of the frame shows
  pavement (lane markings, painted arrows, stop bar, turn bays) and the
  upper half shows overhead features (mast arm, signal heads, lane-use
  panels, regulatory signs such as 'No U-Turn'). Pavement arrows are
  commonly painted within the last 10-20m before the stop bar - in the
  middle-lower region of the image, not the immediate foreground. Scan
  every lane's full length from the camera to the stop bar before
  concluding no arrow is present.""",
)



# ---------- Data models ----------

# Fixed vocabulary of lane-usage tags. Keep in sync with the prompt template.
# bike_lane removed from per-lane vocabulary (reported separately in
# `bike_lane_present` side channel).
LANE_USAGE_TAGS = frozenset({
    "through",
    "left_turn_bay",
    "right_turn_bay",
    "shared_through_left",
    "shared_through_right",
    "u_turn",
    "bus_transit",
    "parking_loading",
    "lane_drops",
    "unknown",
})

# Which tags count as "carries through traffic" for derived lane-count logic.
_THROUGH_CARRYING_TAGS = frozenset({
    "through", "shared_through_left", "shared_through_right",
})


@dataclass(frozen=True)
class PerLaneUsage:
    """VLM's classification of one lane's usage + rationale + confidence."""
    lane: int
    usages: tuple                # tuple[str, ...] — one or more LANE_USAGE_TAGS
    confidence: str              # "high" | "medium" | "low"
    based_on: str                # "observed" | "inferred" | "mixed"
    rationale: str               # one-sentence explanation

    @property
    def carries_through(self) -> bool:
        return any(u in _THROUGH_CARRYING_TAGS for u in self.usages)

    @property
    def is_turn_only(self) -> bool:
        return any(u in ("left_turn_bay", "right_turn_bay") for u in self.usages) and not self.carries_through


@dataclass
class ApproachLaneAuditResult:
    """Per-approach output of a VLM per-lane audit with reasoning support.

    `per_lane` length == observed_lane_count (may differ from declared_lane_count
    when the VLM sees more or fewer lanes than OSM/auto-gen claims).
    """
    sample_label: str
    approach_direction: str
    declared_lane_count: int
    per_lane: list                            # list[PerLaneUsage]
    effective_through_lane_count: Optional[int]
    bike_lane_present: Optional[bool]
    restrictions_visible: list
    image_quality: str
    overall_confidence: str                   # "high" | "medium" | "low"
    raw_response: dict
    observed_lane_count: Optional[int] = None            # VLM's own count; may be != declared
    declared_lane_count_mismatch: str = "match"          # "higher" | "lower" | "match"
    lane_count_mismatch_rationale: str = ""

    def motor_vehicle_lane_count_from_per_lane(self) -> int:
        """Fallback: count per_lane entries NOT solely classified as parking_loading / unknown."""
        non_vehicle = {"parking_loading", "unknown"}
        count = 0
        for pl in self.per_lane:
            if pl.usages and all(u in non_vehicle for u in pl.usages):
                continue
            count += 1
        return count

    @property
    def class_i_candidate(self) -> bool:
        return self.declared_lane_count_mismatch in ("higher", "lower")

    @property
    def suggested_lane_count(self) -> int:
        """What auto-gen's `lanes` should be, per GSV ground truth."""
        if self.observed_lane_count is not None:
            return self.observed_lane_count
        return self.motor_vehicle_lane_count_from_per_lane()


# ---------- CreateAI REST client ----------

class VLMUnavailable(RuntimeError):
    pass


class CreateAIClient:
    """Thin wrapper over ASU CreateAI vision endpoint."""

    def __init__(
        self,
        token: str,
        api_url: str,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        session: Optional[requests.Session] = None,
        model_preset: str = "default",
    ):
        if not token:
            raise ValueError("token is required (got empty)")
        if not api_url:
            raise ValueError("api_url is required")
        self.token = token
        # Allow either base URL (will get /query appended) or full URL (already
        # ending in /query). .env commonly contains the full URL.
        api_url = api_url.rstrip("/")
        if api_url.endswith("/query"):
            self.endpoint_url = api_url
        else:
            self.endpoint_url = api_url + "/query"
        self.model_name, self.model_provider = resolve_model(
            model_name, model_provider, preset=model_preset,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def query_vision(
        self,
        query: str,
        image_bytes: bytes,
        system_prompt: Optional[str] = None,
        mime_type: str = "image/jpeg",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict:
        """Low-level: send one image + query, return the full JSON response dict.

        `max_tokens` overrides the CreateAI project default (typically 1024).
        For multi-pass structured JSON output (4 passes × 4 approaches) we need
        ~3000-4000 tokens; 4096 gives comfortable headroom.

        Cache keys include the override protocol version, input, routing and
        generation parameters. Pre-override caches are never reused. Returned
        provider/model metadata must match the requested identity.
        """
        cache_key = self._cache_key(query, image_bytes, system_prompt, max_tokens, temperature)
        cached = self._load_cache(cache_key)
        if cached is not None:
            try:
                validate_model_identity(cached, self.model_name, self.model_provider)
            except ModelRoutingError as exc:
                raise VLMUnavailable(str(exc)) from exc
            log.info("vlm cache hit: key=%s", cache_key)
            return cached

        encoded = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
        model_params: dict = {
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            model_params["system_prompt"] = system_prompt
        # Service tokens require this flag to override the project's model and
        # parameters for this request. It does not change project settings.
        payload: dict = {
            "endpoint":        "vision",
            "request_source":  REQUEST_SOURCE,
            "query":           query,
            "image_file":      encoded,
            "model_provider":  self.model_provider,
            "model_name":      self.model_name,
            "enable_history":  False,
            "response_format": {"type": "json"},
            "model_params":    model_params,
            "max_tokens":      max_tokens,
        }

        data = self._post_with_retry(self.endpoint_url, payload)
        try:
            validate_model_identity(data, self.model_name, self.model_provider)
        except ModelRoutingError as exc:
            raise VLMUnavailable(str(exc)) from exc
        self._save_cache(cache_key, data)
        return data

    def audit_approach(
        self,
        pavement_jpg: Path,
        signs_jpg: Path,
        intersection_name: str,
        approach_direction: str,
        num_lanes: int,
        sample_location_m: float,
    ) -> ApproachLaneAuditResult:
        """High-level: stitch pavement+signs, build per-lane enumeration prompt, parse.

        Returns an ApproachLaneAuditResult describing what each lane is being used
        for. Downstream code can derive Class I corrections (lane count) and
        Class II movement-level checks from the per-lane usage list without
        needing a candidate list up front.
        """
        stitched = _stitch_vertical(pavement_jpg, signs_jpg)
        prompt = _VERIFICATION_PROMPT_TEMPLATE.format(
            intersection_name=intersection_name,
            approach_direction=approach_direction,
            num_lanes=num_lanes,
            sample_location_m=sample_location_m,
        )
        raw = self.query_vision(prompt, stitched, system_prompt=_SYSTEM_PROMPT)
        parsed = _parse_audit_response(raw, declared_lane_count=num_lanes)
        return ApproachLaneAuditResult(
            sample_label=f"{approach_direction}_{sample_location_m:.0f}m",
            approach_direction=approach_direction,
            declared_lane_count=num_lanes,
            per_lane=parsed["per_lane"],
            effective_through_lane_count=parsed["effective_through_lane_count"],
            bike_lane_present=parsed["bike_lane_present"],
            restrictions_visible=parsed["restrictions_visible"],
            image_quality=parsed["image_quality"],
            overall_confidence=parsed["overall_confidence"],
            raw_response=raw,
            observed_lane_count=parsed.get("observed_lane_count"),
            declared_lane_count_mismatch=parsed.get("declared_lane_count_mismatch", "match"),
            lane_count_mismatch_rationale=parsed.get("lane_count_mismatch_rationale", ""),
        )

    def audit_signs_only(
        self,
        image_jpg: Path,
        intersection_name: str,
        approach_direction: str,
        sample_label: str,
    ) -> dict:
        """Lightweight signs-only audit for signal-facing samples.

        Returns a dict with keys:
            sample_label, approach_direction, restrictions_visible (list),
            image_quality, overall_confidence, rationale, raw_response

        Does NOT populate per_lane / observed_lane_count / mismatch fields
        because signal samples are oriented backward from the driver's
        perspective and are not meant for lane counting.
        """
        prompt = _SIGNS_ONLY_PROMPT_TEMPLATE.format(
            intersection_name=intersection_name,
            approach_direction=approach_direction,
        )
        image_bytes = Path(image_jpg).read_bytes()
        raw = self.query_vision(prompt, image_bytes, system_prompt=_SYSTEM_PROMPT)
        body = _extract_model_payload(raw)
        if isinstance(body, str):
            body = _load_json_lenient(body)
        if not isinstance(body, dict):
            body = {}
        # Optional structured sign classification (added by the upgraded
        # signs prompt). Each entry has restriction_kind / turn_type /
        # lane_position / phase / confidence / rationale. Older audits
        # produced by the legacy prompt will not have this field, so
        # downstream code should tolerate its absence.
        sc_raw = body.get("signs_classified") or []
        signs_classified: list = []
        if isinstance(sc_raw, list):
            for s in sc_raw:
                if not isinstance(s, dict):
                    continue
                signs_classified.append({
                    "raw_text":         str(s.get("raw_text", "") or ""),
                    "restriction_kind": _norm_enum(
                        s.get("restriction_kind"),
                        ("movement", "lane_use", "phase", "advisory"),
                        "advisory",
                    ),
                    "turn_type": (s.get("turn_type")
                                  if s.get("turn_type") in
                                     ("uturn","left","right","through","any")
                                  else None),
                    "lane_position": (s.get("lane_position")
                                      if s.get("lane_position") in
                                         ("leftmost","rightmost","curb")
                                      else None),
                    "phase": (s.get("phase")
                              if s.get("phase") in
                                 ("red_only","protected_only","permissive")
                              else None),
                    "confidence": _norm_enum(
                        s.get("confidence"),
                        ("high","medium","low"),
                        "medium",
                    ),
                    "rationale": str(s.get("rationale", "") or ""),
                })
        return {
            "sample_label":          sample_label,
            "approach_direction":    approach_direction,
            "restrictions_visible":  list(body.get("restrictions_visible", []) or []),
            "signs_classified":      signs_classified,
            "image_quality":         _norm_enum(body.get("image_quality"),
                                                ("good","partially_occluded","unclear"), "unclear"),
            "overall_confidence":    _norm_enum(body.get("overall_confidence"),
                                                ("high","medium","low"), "medium"),
            "rationale":             str(body.get("rationale", "") or ""),
            "raw_response":          raw,
        }

    def audit_approach_single_image(
        self,
        image_jpg: Path,
        intersection_name: str,
        approach_direction: str,
        num_lanes: int,
        sample_label: str,
        sample_location_m: float = 0.0,
    ) -> ApproachLaneAuditResult:
        """Audit variant for tile-based single-view input (pitch=0 horizontal).

        Unlike `audit_approach` which stitches a pavement-pitched view over a
        sign-pitched view, this method sends one 1280x1280 horizontal image
        covering both pavement markings (lower frame) and overhead mast arm
        (upper frame). Intended for inputs produced by
        `gsv_tile_client.TileClient.fetch()`.
        """
        prompt = _SINGLE_VIEW_PROMPT_TEMPLATE.format(
            intersection_name=intersection_name,
            approach_direction=approach_direction,
            num_lanes=num_lanes,
            sample_location_m=sample_location_m,
        )
        image_bytes = Path(image_jpg).read_bytes()
        raw = self.query_vision(prompt, image_bytes, system_prompt=_SYSTEM_PROMPT)
        parsed = _parse_audit_response(raw, declared_lane_count=num_lanes)
        return ApproachLaneAuditResult(
            sample_label=sample_label,
            approach_direction=approach_direction,
            declared_lane_count=num_lanes,
            per_lane=parsed["per_lane"],
            effective_through_lane_count=parsed["effective_through_lane_count"],
            bike_lane_present=parsed["bike_lane_present"],
            restrictions_visible=parsed["restrictions_visible"],
            image_quality=parsed["image_quality"],
            overall_confidence=parsed["overall_confidence"],
            raw_response=raw,
            observed_lane_count=parsed.get("observed_lane_count"),
            declared_lane_count_mismatch=parsed.get("declared_lane_count_mismatch", "match"),
            lane_count_mismatch_rationale=parsed.get("lane_count_mismatch_rationale", ""),
        )

    # ---------- HTTP ----------

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        attempt = 0
        backoff = 1.0
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }
        while True:
            try:
                resp = self.session.post(url, json=payload, headers=headers,
                                         timeout=self.timeout_s)
            except requests.RequestException as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise VLMUnavailable(f"network error: {e}") from e
                log.warning("vlm network error (attempt %d/%d): %s",
                            attempt, self.max_retries, e)
                time.sleep(backoff); backoff *= 2
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise VLMUnavailable(
                        f"vlm 200 but non-JSON body: {resp.text[:300]}"
                    ) from e
            if resp.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > self.max_retries:
                    raise VLMUnavailable(
                        f"vlm {resp.status_code} after {attempt} attempts"
                    )
                log.warning("vlm %d (attempt %d/%d); backoff %.1fs",
                            resp.status_code, attempt, self.max_retries, backoff)
                time.sleep(backoff); backoff *= 2
                continue
            raise VLMUnavailable(f"vlm {resp.status_code}: {resp.text[:400]}")

    # ---------- Cache ----------

    def _cache_key(self, query: str, image_bytes: bytes,
                   system_prompt: Optional[str],
                   max_tokens: int = 4096, temperature: float = 0.0) -> str:
        h = hashlib.sha256()
        h.update(REQUEST_PROTOCOL_VERSION.encode())
        h.update(b"\x00")
        h.update(REQUEST_SOURCE.encode())
        h.update(b"\x00")
        h.update(query.encode())
        h.update(b"\x00")
        h.update(image_bytes)
        h.update(b"\x00")
        if system_prompt:
            h.update(system_prompt.encode())
        h.update(b"\x00")
        h.update(self.model_name.encode())
        h.update(b"\x00")
        h.update(self.model_provider.encode())
        h.update(b"\x00")
        h.update(str(max_tokens).encode())
        h.update(b"\x00")
        h.update(repr(float(temperature)).encode())
        return h.hexdigest()[:20]

    def _load_cache(self, key: str):
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists() and p.stat().st_size > 0:
            return json.loads(p.read_text())
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        if self.cache_dir is None:
            return
        p = self.cache_dir / f"{key}.json"
        p.write_text(json.dumps(data, indent=2, default=str))


# ---------- Helpers: image stitch, prompt format, response parse ----------

def _stitch_vertical(top: Path, bottom: Path, jpeg_quality: int = 95,
                     label_top: str = "PAVEMENT (pitch down)",
                     label_bottom: str = "SIGNS (pitch level)") -> bytes:
    """Concatenate two images top-over-bottom, preserving detail; return JPEG bytes.

    Adds a thin label band between the two views so the VLM can't confuse
    which half is which. Uses JPEG quality 95 (minimal loss) since pavement
    arrows are small features where compression artifacts matter.
    """
    from PIL import Image, ImageDraw, ImageFont
    t = Image.open(top).convert("RGB")
    b = Image.open(bottom).convert("RGB")
    w = max(t.width, b.width)
    band_h = 24
    canvas = Image.new("RGB", (w, t.height + band_h + b.height), (0, 0, 0))
    canvas.paste(t, ((w - t.width) // 2, 0))
    canvas.paste(b, ((w - b.width) // 2, t.height + band_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, t.height + 4),
              f"↑ {label_top}   |   ↓ {label_bottom}",
              fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
    return buf.getvalue()


def _parse_audit_response(raw: dict, declared_lane_count: int) -> dict:
    """Extract and validate the per-lane JSON payload the VLM emitted."""
    body = _extract_model_payload(raw)
    if isinstance(body, str):
        body = _load_json_lenient(body)
    if not isinstance(body, dict):
        raise VLMUnavailable(
            f"vlm did not return a JSON object; got {type(body).__name__}: "
            f"{str(body)[:300]}"
        )

    per_lane: list = []
    seen_lanes: set = set()
    for item in body.get("per_lane", []):
        try:
            lane_idx = int(item["lane"])
            raw_usages = item.get("usages", [])
            if isinstance(raw_usages, str):
                raw_usages = [raw_usages]
            usages = tuple(
                str(u).strip().lower().replace(" ", "_")
                for u in raw_usages
                if str(u).strip()
            )
            clean: list = []
            for u in usages:
                if u in LANE_USAGE_TAGS:
                    clean.append(u)
                elif u == "bike_lane":
                    log.warning("lane %d reported as bike_lane inside per_lane — "
                                "bike_lane should be in side channel; demoted to 'unknown'",
                                lane_idx)
                    clean.append("unknown")
                else:
                    log.warning("unknown lane-usage tag %r on lane %d — coerced to 'unknown'",
                                u, lane_idx)
                    clean.append("unknown")
            if not clean:
                clean = ["unknown"]
            per_lane.append(PerLaneUsage(
                lane=lane_idx,
                usages=tuple(clean),
                confidence=_norm_enum(item.get("confidence"), ("high", "medium", "low"), "low"),
                based_on=_norm_enum(item.get("based_on"), ("observed", "inferred", "mixed"), "observed"),
                rationale=str(item.get("rationale") or item.get("evidence") or ""),
            ))
            seen_lanes.add(lane_idx)
        except (KeyError, TypeError, ValueError) as e:
            log.warning("dropping malformed per_lane entry %r: %s", item, e)

    # Derive the observed lane count BEFORE padding to declared_lane_count.
    # VLM may have returned more or fewer lanes than declared — that's the
    # bidirectional Class I signal we want to preserve.
    vlm_observed = body.get("observed_motor_vehicle_lane_count")
    try:
        vlm_observed = int(vlm_observed) if vlm_observed is not None else None
    except (TypeError, ValueError):
        vlm_observed = None
    if vlm_observed is None:
        # Fall back to max lane index reported (if VLM emitted per_lane but not the top-level count).
        vlm_observed = max(seen_lanes) if seen_lanes else None

    # Only pad to declared_lane_count when the VLM reported FEWER lanes than
    # declared AND didn't explicitly flag a "lower" mismatch. If the VLM
    # deliberately reported fewer (because it genuinely sees fewer), we
    # preserve its count rather than padding with placeholders.
    vlm_mismatch_raw = body.get("declared_lane_count_mismatch")
    vlm_mismatch = str(vlm_mismatch_raw).strip().lower() if vlm_mismatch_raw else None
    should_pad_to_declared = (
        vlm_observed is None
        or (vlm_observed < declared_lane_count and vlm_mismatch != "lower")
    )
    if should_pad_to_declared:
        for i in range(1, declared_lane_count + 1):
            if i not in seen_lanes:
                per_lane.append(PerLaneUsage(
                    lane=i,
                    usages=("unknown",),
                    confidence="low",
                    based_on="observed",
                    rationale="(VLM did not report this lane)",
                ))
    per_lane.sort(key=lambda p: p.lane)

    # If vlm_observed still None, derive it from final per_lane length.
    if vlm_observed is None:
        vlm_observed = len(per_lane)

    # Normalize mismatch — derive if missing/invalid.
    if vlm_mismatch not in ("higher", "lower", "match"):
        if vlm_observed is not None and declared_lane_count is not None:
            if declared_lane_count > vlm_observed: vlm_mismatch = "higher"
            elif declared_lane_count < vlm_observed: vlm_mismatch = "lower"
            else: vlm_mismatch = "match"
        else:
            vlm_mismatch = "match"

    vlm_reported_through = body.get("effective_through_lane_count")
    try:
        vlm_reported_through = int(vlm_reported_through) if vlm_reported_through is not None else None
    except (TypeError, ValueError):
        vlm_reported_through = None
    if vlm_reported_through is None:
        vlm_reported_through = sum(1 for pl in per_lane if pl.carries_through)

    bike_present = body.get("bike_lane_present")
    if isinstance(bike_present, str):
        bike_present = bike_present.strip().lower() in ("true", "yes", "1")
    elif bike_present is not None:
        bike_present = bool(bike_present)

    restrictions = list(body.get("restrictions_visible", []) or [])

    return {
        "per_lane":                       per_lane,
        "effective_through_lane_count":   vlm_reported_through,
        "bike_lane_present":              bike_present,
        "restrictions_visible":           restrictions,
        "image_quality":                  str(body.get("image_quality", "unclear")),
        "overall_confidence":             _norm_enum(body.get("overall_confidence"), ("high", "medium", "low"), "medium"),
        "observed_lane_count":            vlm_observed,
        "declared_lane_count_mismatch":   vlm_mismatch,
        "lane_count_mismatch_rationale":  str(body.get("lane_count_mismatch_rationale") or ""),
    }


def _norm_enum(v, valid: tuple, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip().lower()
    return s if s in valid else default


def _extract_model_payload(raw: dict):
    """CreateAI wraps the model output in a response envelope. Find the JSON body.

    Tries several common keys used by CreateAI / typical LLM gateways.
    """
    if not isinstance(raw, dict):
        return raw
    for key in ("response", "result", "output", "data", "content", "answer", "message"):
        if key in raw:
            v = raw[key]
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                try:
                    return _load_json_lenient(v)
                except Exception:
                    continue
    # Last-resort: assume raw IS the model body.
    return raw


def _load_json_lenient(s: str):
    """Parse JSON that may be wrapped in ```json fences or have leading prose."""
    t = s.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    # Trim to the first '{' and last '}' if there's surrounding prose.
    lo = t.find("{")
    hi = t.rfind("}")
    if lo >= 0 and hi > lo:
        t = t[lo:hi + 1]
    return json.loads(t)


# ---------- Cluster-level orchestration ----------

def run_cluster_audit(
    gsv_dir: Path,
    cluster_node_id: int,
    client: "CreateAIClient",
    intersection_name: str = "",
    sample_filter: Optional[str] = None,
) -> list:
    """Walk `gsv_dir/_manifest.json`; per-lane-audit each sample.

    Returns list[ApproachLaneAuditResult]. The audit is per-lane (not per-movement),
    so it does NOT need a movement.csv input — derivation of movement-level
    corrections happens downstream in verdicts_to_spec.py.

    `sample_filter`: if set, only samples whose `label` contains this string are
    audited (useful for single-approach smoke tests before full-cluster spend).
    """
    manifest = json.loads((gsv_dir / "_manifest.json").read_text())

    results = []
    for s in manifest["samples"]:
        if sample_filter and sample_filter not in s["label"]:
            continue
        direction = s["direction"]

        pavement = gsv_dir / s["label"] / "pavement.jpg"
        signs = gsv_dir / s["label"] / "signs.jpg"
        if not pavement.exists() or not signs.exists():
            log.warning("sample %s: missing image files — skipping", s["label"])
            continue

        log.info("per-lane auditing %s (%d lanes)", s["label"], int(s["num_lanes"]))
        try:
            result = client.audit_approach(
                pavement_jpg=pavement,
                signs_jpg=signs,
                intersection_name=intersection_name or f"cluster@node_{cluster_node_id}",
                approach_direction=direction,
                num_lanes=int(s["num_lanes"]),
                sample_location_m=float(s["cumulative_distance_m"]),
            )
            result.sample_label = s["label"]
            results.append(result)
        except VLMUnavailable as e:
            log.error("sample %s: VLM call failed: %s", s["label"], e)

    return results


def run_cluster_tile_audit(
    tile_dir: Path,
    cluster_node_id: int,
    client: "CreateAIClient",
    intersection_name: str = "",
    sample_filter: Optional[str] = None,
    dir_to_num_lanes: Optional[dict] = None,
) -> list:
    """Audit a cluster whose samples were fetched as single 1280x1280 tile
    projections (see gsv_tile_client.fetch_cluster_tile_samples).

    Unlike run_cluster_audit (which expects pavement.jpg + signs.jpg pairs
    per sample subdirectory), this iterates a flat tile directory containing
    one *.jpg per sample plus a _manifest.json from the tile client.

    `dir_to_num_lanes`: optional {direction_string -> declared lane count
    on the cluster-entry link}, passed to the VLM prompt so it can flag
    mismatch against auto-gen. If None, declared_lane_count is set to 0
    and mismatch flagging is skipped.
    """
    manifest_path = tile_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    results: list = []
    for s in manifest:
        label = s["label"]
        if sample_filter and sample_filter not in label:
            continue
        direction = s["direction"]
        jpg_path = tile_dir / f"{label}.jpg"
        if not jpg_path.exists():
            log.warning("sample %s: image file missing, skipping", label)
            continue
        is_signal = label.endswith("_signal")
        if is_signal:
            # Signal samples: only read signs. Per-lane classification on
            # a reverse-heading image would be misleading (the "lanes" in
            # view are the intersection exit, not the approach stop bar).
            log.info("tile-auditing %s  (signs-only)", label)
            try:
                sign_result = client.audit_signs_only(
                    image_jpg=jpg_path,
                    intersection_name=intersection_name or f"cluster@node_{cluster_node_id}",
                    approach_direction=direction,
                    sample_label=label,
                )
                results.append(sign_result)
            except VLMUnavailable as e:
                log.error("sample %s: VLM call failed: %s", label, e)
            continue
        num_lanes = (dir_to_num_lanes or {}).get(direction, 0)
        log.info("tile-auditing %s  decl=%d", label, num_lanes)
        try:
            result = client.audit_approach_single_image(
                image_jpg=jpg_path,
                intersection_name=intersection_name or f"cluster@node_{cluster_node_id}",
                approach_direction=direction,
                num_lanes=num_lanes,
                sample_label=label,
                sample_location_m=float(s.get("cumulative_distance_m", 0)),
            )
            results.append(result)
        except VLMUnavailable as e:
            log.error("sample %s: VLM call failed: %s", label, e)
    return results


# ---------- CLI ----------

def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run VLM audit on all approach samples of one cluster."
    )
    ap.add_argument("--gsv-dir", required=True, type=Path,
                    help="directory containing per-sample subfolders + _manifest.json")
    ap.add_argument("--cluster-node", required=True, type=int,
                    help="node_id of the consolidated cluster being audited")
    ap.add_argument("--intersection-name", default="", type=str)
    ap.add_argument("--sample-filter", default=None, type=str,
                    help="audit only samples whose label contains this substring "
                         "(e.g. 'SB_link287_d008m' to smoke-test a single sample)")
    ap.add_argument("--cache-dir", default="./cache/vlm", type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--env-file", default=".env", type=Path,
                    help="path to .env with CREATEAI_TOKEN / CREATEAI_API_URL / etc.")
    ap.add_argument("--model-preset", choices=tuple(MODEL_PRESETS), default=None,
                    help="default = GPT-6 Astra; backup = Claude 5.0 Opus")
    ap.add_argument("--model-name", default=None,
                    help="override CREATEAI_MODEL_NAME (e.g., claude5_opus, "
                         "geminipro3_1, gpt5_4_pro, gpt5_4_thinking)")
    ap.add_argument("--model-provider", default=None,
                    help="override CREATEAI_MODEL_PROVIDER (openai | aws | gcp-deepmind | asu-air)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    env = _load_env(args.env_file)
    token   = env.get("CREATEAI_TOKEN")    or os.environ.get("CREATEAI_TOKEN")
    api_url = env.get("CREATEAI_API_URL")  or os.environ.get("CREATEAI_API_URL")
    preset = args.model_preset or os.environ.get("CREATEAI_MODEL_PRESET") or env.get("CREATEAI_MODEL_PRESET")
    if preset:
        model, prov = resolve_model(args.model_name, args.model_provider, preset=preset)
    else:
        model, prov = resolve_model(
            args.model_name or env.get("CREATEAI_MODEL_NAME") or os.environ.get("CREATEAI_MODEL_NAME", DEFAULT_MODEL_NAME),
            args.model_provider or (None if args.model_name else env.get("CREATEAI_MODEL_PROVIDER") or os.environ.get("CREATEAI_MODEL_PROVIDER")),
        )
    if not token or not api_url:
        print("ERROR: CREATEAI_TOKEN / CREATEAI_API_URL missing "
              "(checked env vars and .env)", file=__import__("sys").stderr)
        return 2

    client = CreateAIClient(
        token=token, api_url=api_url,
        model_name=model, model_provider=prov,
        cache_dir=args.cache_dir,
    )

    results = run_cluster_audit(
        gsv_dir=args.gsv_dir,
        cluster_node_id=args.cluster_node,
        client=client,
        intersection_name=args.intersection_name,
        sample_filter=args.sample_filter,
    )

    serializable = []
    for r in results:
        serializable.append({
            "sample_label":                  r.sample_label,
            "approach_direction":            r.approach_direction,
            "declared_lane_count":           r.declared_lane_count,
            "effective_through_lane_count":  r.effective_through_lane_count,
            "observed_lane_count":               r.observed_lane_count,
            "declared_lane_count_mismatch":      r.declared_lane_count_mismatch,
            "lane_count_mismatch_rationale":     r.lane_count_mismatch_rationale,
            "suggested_lane_count":              r.suggested_lane_count,
            "observed_motor_vehicle_lane_count_fallback": r.motor_vehicle_lane_count_from_per_lane(),
            "bike_lane_present":             r.bike_lane_present,
            "per_lane":                      [
                {
                    "lane":       pl.lane,
                    "usages":     list(pl.usages),
                    "confidence": pl.confidence,
                    "based_on":   pl.based_on,
                    "rationale":  pl.rationale,
                }
                for pl in r.per_lane
            ],
            "restrictions_visible":          r.restrictions_visible,
            "image_quality":                 r.image_quality,
            "overall_confidence":            r.overall_confidence,
        })
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "cluster_node_id":   args.cluster_node,
        "intersection_name": args.intersection_name,
        "n_samples":         len(serializable),
        "samples":           serializable,
    }, indent=2, default=str))

    print(f"\naudited {len(results)} sample(s); results → {args.out_json}")
    for r in results:
        through = r.effective_through_lane_count
        decl = r.declared_lane_count
        obs = r.observed_lane_count
        mis = r.declared_lane_count_mismatch
        sug = r.suggested_lane_count
        tag = "✓" if mis == "match" else f"⚠ class-I:{mis}  suggested_lanes={sug}"
        bike = " bike:Y" if r.bike_lane_present else ""
        print(f"  [{r.sample_label}] quality={r.image_quality} conf={r.overall_confidence}{bike}  "
              f"declared={decl} observed={obs} through={through}  {tag}")
        if r.lane_count_mismatch_rationale:
            print(f"      mismatch rationale: {r.lane_count_mismatch_rationale}")
        for pl in r.per_lane:
            print(f"      lane {pl.lane}: {'+'.join(pl.usages)}  "
                  f"[{pl.confidence}/{pl.based_on}]  {pl.rationale}")
        if r.restrictions_visible:
            print(f"      signs: {r.restrictions_visible}")

    return 0


def _load_env(p: Path) -> dict:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


if __name__ == "__main__":
    raise SystemExit(_main())
