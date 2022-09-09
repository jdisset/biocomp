import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "react-flow-renderer";
import { theme } from "./shapes.jsx";
import Util from "./util.jsx";

function hasCopyNumber(data) {
  return data.parameters && data.parameters.copy_number >= 0;
}

function SRCNode(props) {
  const circle_radius = 28;
  const center = circle_radius + 2;

  function generateArcs(arc_span = 35, spacing = 10, radius = circle_radius) {
    let arcs = [];
    const nhandles = props.data.output_to.length;
    const total_span = nhandles * arc_span + (nhandles - 1) * spacing;
    const start_angle = 180 - total_span / 2;
    const end_angle = start_angle + total_span;
    for (let i = 0; i < nhandles; i++) {
      arcs.push(
        <path
          d={Util.describeArc(
            center,
            center,
            radius,
            start_angle + i * (arc_span + spacing),
            start_angle + i * (arc_span + spacing) + arc_span
          )}
          stroke={theme.DNAcolor}
          fill="none"
          strokeWidth="4"
        />
      );
    }
    return arcs;
  }

  let arcs = generateArcs();

  const handle_padding = 1;
  function generateHandles(arc_span = 35, spacing = 10, radius = circle_radius) {
    let handles = [];
    let nhandles = props.data.output_to.length;
    let total_span = nhandles * arc_span + (nhandles - 1) * spacing;
    let start_angle = 180 - total_span / 2;
    let end_angle = start_angle + total_span;
    for (let i = 0; i < nhandles; i++) {
      let arc_center_deg = start_angle + i * (arc_span + spacing) + arc_span / 2;
      let arc_center_x = -(handle_padding + radius) * Math.sin((arc_center_deg * Math.PI) / 180);
      let arc_center_y = (handle_padding + radius) * Math.cos((arc_center_deg * Math.PI) / 180);
      handles.push(
        <Handle
          type="source"
          position={Position.Bottom}
          id={i + 1}
          style={{ left: center + arc_center_x, bottom: center + arc_center_y }}
        />
      );
    }
    return handles;
  }
  let handles = generateHandles();

  return (
    <div className="input-node">
      <svg
        width={(circle_radius + 2) * 2}
        height={(circle_radius + 2) * 2}
        viewBox={`0 0 ${circle_radius * 2 + 4} ${circle_radius * 2 + 4}`}
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx={center} cy={center} r={circle_radius + 1.0} fill="white" />
        <circle cx={center} cy={center} r={circle_radius} stroke="#EEEEEE" stroke-width="3" />

        {arcs}

        <circle cx={center} cy={center} r={circle_radius + 1.5} stroke="black" stroke-width="0.5" />
        <text fill="black" fontSize="8" letterSpacing="-0.05em">
          <tspan x={center} y={center + 2} textAnchor="middle">
            {props.data.source_id}
          </tspan>
        </text>
      </svg>
      <Handle type="target" position={Position.Top} style={{ top: -handle_padding }} />
      {handles}
    </div>
  );
}
export default SRCNode;
