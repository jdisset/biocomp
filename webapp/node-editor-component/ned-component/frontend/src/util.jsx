import dagre from "dagre";
import html2canvas from "html2canvas";
import React from "react";

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
class Util {
  static cmap(x) {
    return cmapcolors[Math.max(0, Math.min(NCOLORS - 1, Math.floor(x * (NCOLORS - 1))))];
  }

  static screenCap = (fname) => {
    html2canvas(document.body, { scale: 2.5 }).then((canvas) => {
      const image = canvas.toDataURL("image/png", 1.0);
      this.downloadImage(image, fname);
    });
  };

  static downloadImage = (blob, fileName) => {
    const fakeLink = window.document.createElement("a");
    fakeLink.style = "display:none;";
    fakeLink.download = fileName;

    fakeLink.href = blob;

    document.body.appendChild(fakeLink);
    fakeLink.click();
    document.body.removeChild(fakeLink);

    fakeLink.remove();
  };

  static polarToCartesian = (centerX, centerY, radius, angle) => {
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  };

  static describeArc = (x, y, radius, value) => {
    const v = Math.min(1.0, Math.max(0.0, value));
    const endAngle = Math.PI * v * 2.0;
    var start = this.polarToCartesian(x, y, radius, endAngle);
    var end = this.polarToCartesian(x, y, radius, -Math.PI / 2.0);
    var largeArcFlag = v <= 0.5 ? "0" : "1";
    var d = ["M", start.x, start.y, "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y].join(
      " "
    );

    return d;
  };

  static hasCopyNumber = (data) => {
    return data.parameters && data.parameters.copy_number >= 0;
  };

  static displayCopyNumber = (data, color = "black") => {
    if (this.hasCopyNumber(data)) {
      let cn = data.parameters.copy_number;
      let col = this.cmap(cn);
      let radius = 15;
      let innerRadius = 14;
      return (
        <div class="copy_number">
          <svg version="1.1" viewBox="-20 -20 40 40" width="40" height="40">
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
              d={this.describeArc(0, 0, radius, 0.2)}
              stroke={color}
              fill="none"
              strokeWidth="5"
            />
            <text transform="translate(-8 3)" fill="black">
              <tspan fontFamily="Roboto" fontSize="8" fontWeight="300" fill="black" x="0" y="0">
                0.45
              </tspan>
            </text>
          </svg>
        </div>
      );
    } else return "";
  };

  //{cn.toFixed(1)}
  //strokeWidth={0.5 + Math.max(0, cn * 5.0)}
  static zeroPad = (num, places) => String(num).padStart(places, "0");

  static getLayoutedElements = (
    nodes,
    edges,
    nodeWidth = 150,
    nodeHeight = 270,
    dimensionsDict = {},
    direction = "TB"
  ) => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    const isHorizontal = direction === "LR";
    dagreGraph.setGraph({ rankdir: direction });
    nodes.forEach((node) => {
      const w = node.type in dimensionsDict ? dimensionsDict[node.type].width : nodeWidth;
      const h = node.type in dimensionsDict ? dimensionsDict[node.type].height : nodeHeight;
      dagreGraph.setNode(node.id, { width: w, height: h });
    });
    edges.forEach((edge) => {
      dagreGraph.setEdge(edge.source, edge.target);
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
      return node;
    });
    return { nodes, edges };
  };
}
export default Util;
