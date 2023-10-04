import dagre from "dagre";
import html2canvas from "html2canvas";
import React from "react";
import SEQNode from "./SEQNode.jsx";
import AGGNode from "./AGGNode.jsx";
import SRCNode from "./SRCNode.jsx";
import TLNode from "./TLNode.jsx";
import TCNode from "./TCNode.jsx";
import INNode from "./INNode.jsx";
import OUTNode from "./OUTNode.jsx";
import NUMNode from "./NUMNode.jsx";
import CTENode from "./CTENode.jsx";
import INVNode from "./INVNode.jsx";
import DENDNode from "./DENDNode.jsx";
import ContentEdge from "./ContentEdge.jsx";

let colormap = require("colormap");
const NCOLORS = 100;
const cmapcolors = colormap({
  colormap: [
    { index: 0.0, rgb: [41, 41, 41] },
    { index: 0.33, rgb: [86, 34, 50] },
    { index: 0.67, rgb: [173, 42, 92] },
    { index: 1.0, rgb: [233, 43, 71] },
  ],
  nshades: NCOLORS,
  format: "hex",
  alpha: 1,
});

const computeNodeTypes = {
  sequestron_ERN: SEQNode,
  sequestron_RECOMBINASE: SEQNode,
  translation: TLNode,
  transcription: TCNode,
  bias: CTENode,
  in: INNode,
  out: OUTNode,
  input: INNode,
  output: OUTNode,
  aggregation: AGGNode,
  source: SRCNode,
  numeric: NUMNode,
  deadend: DENDNode,
  inv_numeric: INVNode,
  inv_source: INVNode,
  inv_aggregation: INVNode,
  inv_transcription: INVNode,
  inv_translation: INVNode,
};

const typeDim = {
  sequestron_ERN: { width: 200, height: 60 },
  sequestron_RECOMBINASE: { width: 100, height: 70 },
  translation: { width: 40, height: 80 },
  transcription: { width: 40, height: 80 },
  inv_transcription: { width: 40, height: 5 },
  inv_translation: { width: 40, height: 5 },
  inv_source: { width: 40, height: 5 },
  inv_aggregation: { width: 40, height: 5 },
  source: { width: 30, height: 120 },
  aggregation: { width: 150, height: 30 },
  bias: { width: 45, height: 40 },
  output: { width: 20, height: 50 },
};

class Util {
  static cmap(x) {
    return cmapcolors[Math.max(0, Math.min(NCOLORS - 1, Math.floor(x * (NCOLORS - 1))))];
  }

  static describeArc = (x, y, r, sAng, eAng) => {
    var M = Math;
    eAng = M.max(0.0, eAng - 0.0001);
    var f = eAng - sAng <= 180 ? 0 : 1,
      q,
      cXY = (x, y, a) => {
        q = ((a - 90) * M.PI) / 180;
        return [x + r * M.cos(q), y + r * M.sin(q)];
      };
    return ["M", ...cXY(x, y, eAng), "A", r, r, 0, f, 0, ...cXY(x, y, sAng)].join(" ");
  };

  static displayCopyNumber = (data, color = "black", onMouseDown = null) => {
    const MAX_COPY_N = 1.0;
    if (data !== undefined && data !== null) {
      let cn = data;
      let v = Math.min(1.0, Math.max(0.0, cn / MAX_COPY_N));
      let col = this.cmap(cn);
      let radius = 15;
      let innerRadius = 14;
      return (
        <div className="copy_number">
          <svg
            version="1.1"
            viewBox="-20 -20 40 40"
            width="40"
            height="40"
            onMouseDown={onMouseDown}
          >
            <circle cx="0" cy="0" r={radius} fill="white" strokeWidth="0.25" stroke="black" />
            <circle
              cx="0"
              cy="0"
              r={innerRadius}
              fill="white"
              strokeDasharray={"0.25," + ((Math.PI * innerRadius * 2.0) / 8 - 0.5) + ",0.25,0"}
              strokeWidth="2"
              stroke="black"
            />

            <path
              d={this.describeArc(0, 0, radius, 0, 360.0 * v)}
              stroke={color}
              fill="none"
              strokeWidth="5"
            />

            <text
              transform="translate(0 1)"
              fill="black"
              dominantBaseline="middle"
              textAnchor="middle"
            >
              <tspan fontFamily="Roboto" fontSize="10" fontWeight="300" fill="black" x="0" y="0">
                {cn.toFixed(1)}
              </tspan>
            </text>
          </svg>
        </div>
      );
    } else return "";
  };

  static zeroPad = (num, places) => String(num).padStart(places, "0");

  static getLayoutedElements = (
    dagreGraph,
    nodes,
    edges,
    nodeWidth = 150,
    nodeHeight = 270,
    dimensionsDict = {},
    direction = "TB",
  ) => {
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    const isHorizontal = direction === "LR";
    dagreGraph.setGraph({ rankdir: direction, multigraph: true, compound: true });
    nodes.forEach((node) => {
      const w = node.type in dimensionsDict ? dimensionsDict[node.type].width : nodeWidth;
      const h = node.type in dimensionsDict ? dimensionsDict[node.type].height : nodeHeight;
      dagreGraph.setNode(node.id, { width: w, height: h });
    });
    edges.forEach((edge) => {
      var target = JSON.parse(edge.target);
      if (Array.isArray(target)) {
        target.forEach(function (item) {
          dagreGraph.setEdge(edge.source, item);
        });
      } else {
        dagreGraph.setEdge(edge.source, edge.target);
      }
    });
    dagre.layout(dagreGraph);
    nodes.forEach((node) => {
      const w = node.type in dimensionsDict ? dimensionsDict[node.type].width : nodeWidth;
      const h = node.type in dimensionsDict ? dimensionsDict[node.type].height : nodeHeight;
      const nodeWithPosition = dagreGraph.node(node.id);
      node.targetPosition = isHorizontal ? "left" : "top";
      node.sourcePosition = isHorizontal ? "right" : "bottom";
      // We are shifting the dagre node position (anchor=center center) to the top left
      // so it matches the React Flow node anchor point (top left).
      node.position = {
        x: nodeWithPosition.x - w / 2,
        y: nodeWithPosition.y - h / 2,
      };

      if (node.type == "aggregation" || node.type == "bias") {
        node.dragHandle = ".drag-handle";
      }

      return node;
    });
    return { nodes, edges };
  };

  static fillEdgesWithNodeData = (edges, nodes) =>
    edges.map((e) => {
      const src_id = parseInt(e.data.source_node_list_id);
      const tgt_id = parseInt(e.data.target_node_list_id);
      const src_node = nodes[src_id];
      const tgt_node = nodes[tgt_id];
      return {
        ...e,
        data: {
          ...e.data,
          srcdata: src_node.data,
          tgtdata: tgt_node.data,
        },
      };
    });
}

export { Util as default, computeNodeTypes, typeDim };
