"""army-of-loras-comfyui: route prompts to an army of community LoRAs with pybed embeddings."""

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

if __package__:  # loaded by ComfyUI as a custom-node package (not e.g. by pytest's collector)
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F811
    from . import server_routes  # noqa: F401  (registers /army/* HTTP routes)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
