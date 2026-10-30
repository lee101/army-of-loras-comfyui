"""HTTP routes: /army/route (live routing JSON), /army/catalog, /army/studio (one-page app)."""

from __future__ import annotations

import sys
from pathlib import Path

from aiohttp import web
from server import PromptServer

from .army import router, safety

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
from build_workflows import BRANCHES, build  # noqa: E402

routes = PromptServer.instance.routes


@routes.get("/army/route")
async def army_route(request: web.Request) -> web.Response:
    q = request.query
    family = q.get("family", "auto")
    mode = "auto"
    if family in ("auto_image", "auto_video"):
        mode, family = family.split("_", 1)[1], "auto"
    try:
        plan = router.route(q.get("prompt", ""), family=family, mode=mode,
                            max_loras=int(q.get("max_loras", 3)))
    except safety.UnsafePromptError as e:
        return web.json_response({"refused": str(e)}, status=200)
    return web.json_response({**plan.to_dict(), "report": plan.report()})


@routes.get("/army/catalog")
async def army_catalog(request: web.Request) -> web.Response:
    out = {}
    for fam, cat in router.all_catalogs().items():
        out[fam] = {"label": cat["label"], "mode": cat["mode"], "base": cat["base"],
                    "base_license": cat["base_license"],
                    "loras": [{k: l.get(k) for k in ("id", "name", "repo", "file", "license", "group", "trigger")}
                              for l in cat["loras"]]}
    return web.json_response(out)


@routes.get("/army/studio")
async def army_studio(request: web.Request) -> web.Response:
    return web.FileResponse(HERE / "web" / "studio.html")


@routes.get("/army/workflow/{family}")
async def army_workflow(request: web.Request) -> web.Response:
    """API-format workflow for a family (or 'auto'), with this machine's models.local.json applied."""
    family = request.match_info["family"].removesuffix(".json").removeprefix("army_")
    if family != "auto" and family not in BRANCHES:
        raise web.HTTPNotFound()
    return web.json_response(build(family))
