"""Design-guide criteria per facility type, evaluated as far as the network can show them.

Status values
  met         verified on the network topology or control coding
  not_met     the network contradicts the criterion
  schematic   a quantity measured on the schematic geometry; indicative only
  measured    a quantity measured on real geometry (OSM via the Interchange Atlas)
  needs_data  needs lane-level design, signal timing or demand that the network does not carry
  info        context for the reader, nothing to verify

Source order: Arizona (ADOT) first, then national documents (NCHRP, AASHTO, FHWA). Another state's
guide is used only where neither gives the figure. Criteria quote or paraphrase the cited section;
ADOT RDG and NCHRP 959 were read on 2026-10-07, the others on 2026-10-05. Geometry here is
schematic, so no geometric criterion can be certified.
"""
from __future__ import annotations

import math

from .checks import group_links, links_cross, shortest_path

ADOT_RDG_URL = "https://azdot.gov/sites/default/files/2026-04/2021-RDG-with-Jan-2025-revisions.pdf"

# key: (title, url, level, short name)
SOURCES = {
    "adot_rdg": ("ADOT Roadway Design Guidelines, 2021 ed. with Jan. 2025 revisions", ADOT_RDG_URL, "Arizona", "ADOT RDG"),
    "nchrp959": ("NCHRP Research Report 959: Diverging Diamond Interchange Informational Guide, 2nd ed., 2021",
                 "https://nap.nationalacademies.org/26027", "National", "NCHRP 959"),
    "nchrp345": ("NCHRP Report 345: Single Point Urban Interchange Design and Operations Analysis, 1991",
                 "https://trid.trb.org/View/364853", "National", "NCHRP 345"),
    "nchrp1043": ("NCHRP Research Report 1043: Guide for Roundabouts, 2023",
                  "https://nap.nationalacademies.org/catalog/27069/guide-for-roundabouts", "National", "NCHRP 1043"),
    "aashto_gb": ("AASHTO A Policy on Geometric Design of Highways and Streets (Green Book), 2018", None, "National", "AASHTO Green Book"),
    "fhwa_tat14": ("FHWA Traffic Analysis Toolbox Vol. XIV (FHWA-HOP-13-015)",
                   "https://ops.fhwa.dot.gov/publications/fhwahop13015/sec7.htm", "National", "FHWA TAT XIV"),
    "caltrans_dib92": ("Caltrans Design Information Bulletin 92: Single Point Interchange Guidelines, 2018",
                       "https://dot.ca.gov/-/media/dot-media/programs/design/documents/f0005291-dib92-a11y.pdf", "Other state", "Caltrans DIB 92"),
}


# printed page -> page of the ADOT RDG PDF, so a link opens at the cited page
ADOT_PDF_PAGE = {"500-4": 215, "500-5": 216, "500-30": 241, "500-37": 248, "500-39": 250}


def _c(cid, text, src, where, kind, status, detail, value=None):
    title, doc_url, level, short = SOURCES[src]
    url = doc_url
    if src == "adot_rdg" and where:
        url = f"{doc_url}#page={ADOT_PDF_PAGE[where.rsplit('p. ', 1)[1]]}"
    return {"id": cid, "criterion": text, "source": f"{title}, {where}" if where else title, "url": url,
            "doc": title, "doc_url": doc_url, "doc_short": short, "short": f"{short} {where}" if where else short,
            "level": level, "kind": kind, "status": status, "value": value, "detail": detail}


def _passed(checks, *names):
    by = {c["check"].split()[0]: c["passed"] for c in checks}
    return all(by.get(n, False) for n in names)


def _left_turn_conflicts(net, terminals):
    """Per terminal: does the cross-street left turn onto the on-ramp cross the opposing through movement?"""
    out = {}
    for t in terminals:
        allowed = group_links(net, t.group)
        left = shortest_path(net, t.ports["bridge_in"], t.ports["onramp_out"], allowed)
        thru = shortest_path(net, t.ports["approach_in"], t.ports["bridge_out"], allowed)
        if not left or not thru:
            out[t.name] = None
            continue
        out[t.name] = any(links_cross(net.links[a], net.links[b]) for a in left[0] for b in thru[0] if a != b)
    return out


def _crossover_angle(net, t):
    a, b = net.xy(t.ports["approach_in"]), net.xy(t.ports["bridge_out"])
    c, d = net.xy(t.ports["bridge_in"]), net.xy(t.ports["departure_out"])
    v1, v2 = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
    cos = abs(v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return round(math.degrees(math.acos(min(1.0, cos))), 1)


def _skew(site):
    if not site or site.get("freeway_bearing_deg") is None:
        return None
    d = abs(site["freeway_bearing_deg"] - site["crossroad_bearing_deg"]) % 180
    return round(min(d, 180 - d), 1)


def _left_turn_signalled(net, terminals):
    """Per terminal: is any link on the cross-street left turn onto the on-ramp coded as signal-controlled?"""
    out = {}
    for t in terminals:
        path = shortest_path(net, t.ports["bridge_in"], t.ports["onramp_out"], group_links(net, t.group))
        out[t.name] = None if not path else any(
            str(net.links[l].get("control_status", "")).startswith("signal_required") for l in path[0])
    return out


def evaluate(target: str, base, after, terminals, spec, checks, site=None) -> list[dict]:
    out = []
    ramp_angle = ("Each exit and entrance ramp meets the crossroad as close to a right angle as practical, "
                  "and never more than 15 degrees from it")
    ramp_angle_detail = ("Ramps are drawn as straight lines from the freeway to the ramp terminal, "
                         "so the angle where they meet the crossroad is not represented")
    if target == "diamond":
        conf = _left_turn_conflicts(after, terminals)
        out.append(_c("DIA-1", "A compact diamond loses efficiency as left-turn demand at the ramp terminals grows; "
                      "the ramp/crossroad intersections limit the capacity of the interchange",
                      "adot_rdg", "§502.2 A, p. 500-4", "operations", "info",
                      "Existing interchange; turning volumes are not in the network"))
        out.append(_c("DIA-2", "Cross-street left turns onto the ramps cross the path of opposing through traffic "
                      "(the conflict a diverging diamond removes)",
                      "nchrp959", "§1.5, pp. 2-3", "topology", "info",
                      ", ".join(f"{k}: {'crosses' if v else 'no crossing'}" for k, v in conf.items()),
                      value=sum(bool(v) for v in conf.values())))
        out.append(_c("DIA-3", ramp_angle, "adot_rdg", "§505.1 B, p. 500-30", "geometry", "needs_data", ramp_angle_detail))

    if target == "ddi":
        out.append(_c("DDI-S1", "Cross-street traffic moves to the left side of the road between the signalized ramp intersections",
                      "adot_rdg", "§502.2 F, p. 500-5", "topology", "met" if _passed(checks, "DDI-1", "DDI-2") else "not_met",
                      "Confirmed by the network checks: through traffic crosses over at both terminals and drives on the left between them"))
        conf = _left_turn_conflicts(after, terminals)
        sig = _left_turn_signalled(after, terminals)
        ok = all(v is False for v in conf.values()) and all(v is False for v in sig.values())
        out.append(_c("DDI-S2", "Cross-street left turns continue to the ramps without conflicting with opposing through traffic "
                      "and without stopping",
                      "adot_rdg", "§502.2 F, p. 500-5", "topology", "met" if ok else "not_met",
                      "; ".join(f"{k}: {'crosses opposing through' if conf[k] else 'no crossing'}, "
                                f"{'signal on the path' if sig.get(k) else 'no signal on the path'}" for k in conf)))
        xs = [n for n in after.nodes.values() if n["node_type"] == "crossover"]
        shaped = _passed(checks, "DDI-3") and all(n["ctrl_type"] == "signal" for n in xs)
        out.append(_c("DDI-S3", "Two-phase signal operation",
                      "adot_rdg", "§502.2 F, p. 500-5", "operations", "met" if shaped else "not_met",
                      f"{len(xs)} crossovers, each with a single approach per direction and coded as a signal, so two phases "
                      "are enough; phase lengths are not in the network"))
        angles = {t.name: _crossover_angle(after, t) for t in terminals}
        out.append(_c("DDI-S4", "Provide the largest crossover angle the site allows; early DDI practice targeted 45 degrees or more",
                      "nchrp959", "§6.4.7.3, p. 148", "geometry", "schematic",
                      ", ".join(f"{k}: {v}°" for k, v in angles.items())
                      + (" on the drawing (45° or more)" if all(a >= 45 for a in angles.values()) else " on the drawing (below 45°)"),
                      value=min(angles.values())))
        out.append(_c("DDI-S5", "Crossover target speeds typically 20-35 mph, with curve radii of 100-400 ft to match",
                      "nchrp959", "§6.4.7.1, p. 147", "geometry", "needs_data",
                      "Intersection connectors are drawn as straight lines, not lane-by-lane alignments"))
        out.append(_c("DDI-S6", "Most applicable where left turns onto the ramps are heavy, or left turns from the ramps are moderate to heavy",
                      "adot_rdg", "§502.2 F, p. 500-5", "operations", "needs_data", "Needs turning volumes; the network has none"))
        out.append(_c("DDI-S7", ramp_angle, "adot_rdg", "§505.1 B, p. 500-30", "geometry", "needs_data", ramp_angle_detail))

    if target == "spui":
        out.append(_c("SPUI-S1", "The ramps and the crossroad meet at one intersection in the centre of the interchange; "
                      "inside left turns reduce the number of signal phases",
                      "adot_rdg", "§502.2 C, p. 500-4", "topology", "met" if _passed(checks, "SPUI-1", "SPUI-2") else "not_met",
                      "Confirmed by the network checks: one signal group, and each movement crosses the intersection once"))
        out.append(_c("SPUI-S2", "Frontage-road through traffic needs extra signal phases and lowers the efficiency of a SPUI",
                      "adot_rdg", "§502.2 C, p. 500-4", "operations", "info",
                      "The SR 347 / I-10 network has no frontage roads; signal phasing and timing are not in the network"))
        out.append(_c("SPUI-S3", "Turning paths: at least 6 ft (desirably 10 ft) between opposing left turns; dual left-turn lanes "
                      "15-16 ft wide; compound curves with a radius ratio of no more than 2:1",
                      "adot_rdg", "§505.2, p. 500-37", "geometry", "needs_data",
                      "Needs lane-level turning paths; intersection connectors are drawn as straight lines"))
        out.append(_c("SPUI-S4", "Keep crest vertical curvature across the intersection small so drivers can see from stop bar to stop bar",
                      "adot_rdg", "§505.2, p. 500-37", "geometry", "needs_data", "The network has no vertical profile"))
        out.append(_c("SPUI-S5", "Locate pedestrian crossings of the crossroad away from the SPUI, at the nearest cross street",
                      "adot_rdg", "§505.2, p. 500-37", "multimodal", "needs_data", "The network has no pedestrian facilities"))
        skew = _skew(site)
        out.append(_c("SPUI-S6", "Freeway and crossroad should intersect at no less than 75 degrees (below 60 only in extreme cases)",
                      "caltrans_dib92", None, "geometry", "measured" if skew is not None else "needs_data",
                      ((f"{skew}° between I-10 and Queen Creek Rd, measured on OpenStreetMap"
                        + (" (at least 75°)" if skew >= 75 else " (below 75°)")) if skew is not None else "No real road directions available")
                      + ". No Arizona or national figure was found, so the California value is used",
                      value=skew))
        out.append(_c("SPUI-S7", "Weigh the higher cost (more right-of-way; a bridge without a centre pier) against the greater operational efficiency",
                      "adot_rdg", "§502.2 C, p. 500-4", "process", "info",
                      "Costs are not in the network; the existing diamond and the diverging diamond are on this page for comparison"))
        out.append(_c("SPUI-S8", "SPUI design and operations data from 36 sites; ADOT RDG §505.2 refers to it for further guidance",
                      "nchrp345", None, "info", "info", "Background reference"))

    if target == "roundabout_diamond":
        out.append(_c("RBT-S1", "One-way counter-clockwise circulation; entering traffic yields to circulating traffic",
                      "nchrp1043", None, "topology", "met" if _passed(checks, "RBT-1", "RBT-2", "RBT-3") else "not_met",
                      "Confirmed by the network checks"))
        out.append(_c("RBT-S2", "Every entry serves every exit, U-turns included",
                      "nchrp1043", None, "topology", "met" if _passed(checks, "C3") else "not_met", "Confirmed by the turning-movement check"))
        rings = [n for n in after.nodes.values() if n["node_type"] == "roundabout"]
        icd = None
        if rings:
            g = rings[0]["control_group"]
            pts = [after.xy(n["node_id"]) for n in rings if n["control_group"] == g]
            cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
            icd = round(2 * sum(math.dist((cx, cy), p) for p in pts) / len(pts), 1)
        out.append(_c("RBT-S3", "Size the inscribed circle from the design vehicle and fastest-path speeds (performance-based design)",
                      "nchrp1043", None, "geometry", "needs_data",
                      f"Drawn ring diameter {icd} m ({icd * 3.281:.0f} ft); lanes, design vehicle and speeds are not in the network"
                      if icd else "no ring", value=icd))

    if target != "diamond":
        out.append(_c("GEN-1", "Network error checks: missing ramp connections, floating links, turn prohibitions, grade separation",
                      "fhwa_tat14", "ch. 7.1", "topology", "met" if _passed(checks, "C1", "C2", "C3", "C4", "C6") else "not_met",
                      "Confirmed by the network checks; lane-count continuity cannot be checked because lane counts are blank"))
    out.append(_c("GEN-2", "Number of crossroad lanes at the ramp intersections from design-year through and turning volumes, "
                  "compatible with the local agency's typical section",
                  "adot_rdg", "§505.3, p. 500-39", "geometry", "needs_data", "Lane counts and traffic volumes are blank in the network"))
    out.append(_c("GEN-3", "Interchange geometric design (ADOT adopts the 2018 Green Book as a companion to its guidelines)",
                  "aashto_gb", "ch. 10", "geometry", "needs_data", "Not checked: the network is a drawing without real coordinates"))
    return out
