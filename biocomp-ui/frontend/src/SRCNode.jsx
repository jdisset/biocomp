import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "react-flow-renderer";
import { theme } from "./shapes.jsx";
import Util from "./util.jsx";

function hasCopyNumber(data) {
  return data.parameters && data.parameters.copy_number >= 0;
}

function SRCNode(props) {
  console.log(props);
  function generateHandles() {
    let handles = [];
    let nhandles = props.data.output_to.length;
    for (let i = 0; i < nhandles; i++) {
      let spacing = 5;
      let total_length = spacing * (nhandles - 1);
      handles.push(
        <Handle
          type="source"
          position={Position.Bottom}
          id={i+1}
          style={{ bottom: 0, left: 25 - total_length / 2 + i * spacing }}
        />
      );
    }
    return handles;
  }

  let handles = generateHandles();
  return (
    <div className="input-node">
      <svg
        width="51"
        height="51"
        viewBox="0 0 51 51"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle
          cx="25.5877"
          cy="25.7267"
          r="24.75"
          stroke="black"
          stroke-width="0.5"
          fill="white"
        />
        <circle cx="25.5877" cy="25.7267" r="24" stroke={theme.DNAcolor} stroke-width="2" />
        <circle cx="25.5877" cy="25.7267" r="25" stroke="black" stroke-width="0.5" />
        <text fill="black" fontSize="6" letterSpacing="0em">
          <tspan x="26" y="28" textAnchor="middle">
            {props.data.source_id}
          </tspan>
        </text>
      </svg>
      {Util.displayCopyNumber(props.data)}
      <Handle type="target" position={Position.Top} style={{ top: 0 }} />
      {handles}
    </div>
  );
}
export default SRCNode;
