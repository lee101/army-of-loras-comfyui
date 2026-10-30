// Shows the routing report produced by the Router / Auto Output nodes right on the node.
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODES = new Set(["ArmyLoraRouter", "ArmyAutoOutput"]);

app.registerExtension({
  name: "army-of-loras.report",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.has(nodeData.name)) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.text?.join("\n");
      if (!text) return;
      let w = this.widgets?.find((w) => w.name === "army_report");
      if (!w) {
        w = ComfyWidgets.STRING(this, "army_report", ["STRING", { multiline: true }], app).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.fontFamily = "monospace";
        w.inputEl.style.fontSize = "11px";
        w.serialize = false;
      }
      w.value = text;
      this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], 360)]);
      app.graph.setDirtyCanvas(true, true);
    };
  },
});
