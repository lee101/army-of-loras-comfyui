"""Convert an API-format workflow into a UI (LiteGraph) workflow you can drag into ComfyUI.

Slot names/types and widget order come from a running server's /object_info, so the result
matches the installed node versions. Nodes are laid out left-to-right by dependency depth,
one horizontal lane (and labelled group) per model family.
"""

from __future__ import annotations

import json
import urllib.request

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
LANE_TITLES = {
    "router": "Army of LoRAs: pybed router + auto output",
    "flux2_klein9b": "FLUX.2 [klein] 9B  (image)",
    "qwen_image21": "Qwen-Image 2.1  (image)",
    "moss": "MOSS-Image  (image)",
    "ltx23": "LTX-2.3  (video + audio)",
    "h3": "MiniMax H3  (video + audio)",
}
LANE_COLORS = {"router": "#3f789e", "flux2_klein9b": "#8A8", "qwen_image21": "#88A", "moss": "#a1309b",
               "ltx23": "#b58b2a", "h3": "#b06634"}


def _is_widget(spec) -> bool:
    t = spec[0]
    return isinstance(t, list) or t in WIDGET_TYPES


def _inputs(info: dict) -> list[tuple[str, list, bool]]:
    order = info.get("input_order") or {k: list(v) for k, v in info["input"].items()}
    out = []
    for section in ("required", "optional"):
        specs = info["input"].get(section, {})
        for name in order.get(section, specs.keys()):
            if name in specs:
                out.append((name, specs[name], section == "optional"))
    return out


def convert(api: dict, lanes: dict[str, str], server: str) -> dict:
    if server.startswith("http"):
        with urllib.request.urlopen(f"{server}/object_info", timeout=60) as r:
            object_info = json.load(r)
    else:  # a saved copy of /object_info
        with open(server) as f:
            object_info = json.load(f)

    ids = {k: i + 1 for i, k in enumerate(api)}
    links, nodes = [], []
    out_links: dict[tuple[int, int], list[int]] = {}

    # depth = longest path from a source, for x placement
    depth: dict[str, int] = {}

    def d(k):
        if k not in depth:
            depth[k] = 0
            deps = [v[0] for v in api[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
            depth[k] = 1 + max((d(x) for x in deps), default=-1)
        return depth[k]

    for k in api:
        d(k)

    lane_order = [l for l in LANE_TITLES if l in set(lanes.values())]
    lane_y, y = {}, 0
    col_y: dict[tuple[str, int], int] = {}
    heights: dict[str, int] = {}
    for k, node in api.items():
        info = object_info[node["class_type"]]
        ins = _inputs(info)
        n_widgets = sum(_is_widget(spec) for _, spec, _ in ins)
        n_sockets = max(len(ins) - n_widgets, len(info["output"]))
        multiline = any(isinstance(sp[1], dict) and sp[1].get("multiline") for _, sp, _ in ins if len(sp) > 1)
        heights[k] = 60 + 24 * n_sockets + 30 * n_widgets + (140 if multiline else 0)
    for lane in lane_order:
        members = [k for k in api if lanes.get(k) == lane]
        cols: dict[int, int] = {}
        for k in members:
            cols[depth[k]] = cols.get(depth[k], 0) + heights[k] + 50
        lane_y[lane] = y
        y += max(cols.values(), default=200) + 120

    link_id = 0
    for k, node in api.items():
        info = object_info[node["class_type"]]
        nid = ids[k]
        inputs, widgets = [], []
        for name, spec, _ in _inputs(info):
            val = node["inputs"].get(name)
            linked = isinstance(val, list) and len(val) == 2 and isinstance(val[0], str)
            if _is_widget(spec):
                default = spec[1].get("default") if len(spec) > 1 and isinstance(spec[1], dict) else None
                if isinstance(spec[0], list) and default is None:
                    default = spec[0][0] if spec[0] else ""
                widgets.append(default if linked or val is None else val)
                if len(spec) > 1 and isinstance(spec[1], dict) and (spec[1].get("control_after_generate")):
                    widgets.append("fixed")
                if linked:
                    link_id += 1
                    src, slot = ids[val[0]], val[1]
                    t = "STRING" if isinstance(spec[0], list) else spec[0]
                    links.append([link_id, src, slot, nid, len(inputs), t])
                    out_links.setdefault((src, slot), []).append(link_id)
                    inputs.append({"name": name, "type": t, "widget": {"name": name}, "link": link_id})
            else:
                t = spec[0]
                entry = {"name": name, "type": t, "link": None}
                if linked:
                    link_id += 1
                    src, slot = ids[val[0]], val[1]
                    links.append([link_id, src, slot, nid, len(inputs), t])
                    out_links.setdefault((src, slot), []).append(link_id)
                    entry["link"] = link_id
                inputs.append(entry)
        lane = lanes.get(k, "router")
        col = depth[k]
        cy = col_y.get((lane, col), lane_y[lane] + 60)
        col_y[(lane, col)] = cy + heights[k] + 50
        nodes.append({
            "id": nid, "type": node["class_type"], "pos": [col * 440, cy],
            "size": [380, heights[k]], "flags": {}, "order": 0, "mode": 0,
            "inputs": inputs,
            "outputs": [{"name": (info.get("output_name") or info["output"])[i], "type": t, "links": [],
                         "slot_index": i} for i, t in enumerate(info["output"])],
            "properties": {"Node name for S&R": node["class_type"]},
            "widgets_values": widgets,
            **({"title": node["_meta"]["title"]} if node.get("_meta", {}).get("title") else {}),
            **({"color": "#223", "bgcolor": "#335"} if node["class_type"].startswith("Army") else {}),
        })
    by_id = {n["id"]: n for n in nodes}
    for (src, slot), lids in out_links.items():
        by_id[src]["outputs"][slot]["links"] = lids
    for i, n in enumerate(sorted(nodes, key=lambda n: depth[next(k for k, v in ids.items() if v == n["id"])])):
        n["order"] = i

    groups = []
    for lane in lane_order:
        members = [by_id[ids[k]] for k in api if lanes.get(k) == lane]
        x0 = min(n["pos"][0] for n in members) - 20
        y0 = min(n["pos"][1] for n in members) - 50
        x1 = max(n["pos"][0] + n["size"][0] for n in members) + 20
        y1 = max(n["pos"][1] + n["size"][1] for n in members) + 20
        groups.append({"title": LANE_TITLES[lane], "bounding": [x0, y0, x1 - x0, y1 - y0],
                       "color": LANE_COLORS[lane], "font_size": 28, "flags": {}})
    return {"last_node_id": len(nodes), "last_link_id": link_id, "nodes": nodes, "links": links,
            "groups": groups, "config": {}, "extra": {"ds": {"scale": 0.5, "offset": [40, 40]}}, "version": 0.4}
