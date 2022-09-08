import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "react-flow-renderer";
import { theme } from "./shapes.jsx";
import Util from "./util.jsx";

function hasCopyNumber(data) {
  return data.parameters && data.parameters.copy_number >= 0;
}

function AGGNode(props) {
  console.log(props);
  let nCircles = props.data.output_to.length;
  let circleRadius = 14;
  let circleSpacing = 8;

  function generateCircles() {
    let circles = [];
    for (let i = 0; i < nCircles; i++) {
      let x = circleRadius + i * (circleRadius * 2 + circleSpacing);
      let y = circleRadius;
      circles.push(
        <circle cx={x} cy={y} r={circleRadius - 2} stroke="#DDDDDD" stroke-width="3.5" />
      );
      circles.push(
        <circle cx={x} cy={y} r={circleRadius - 0.5} stroke="black" stroke-width="0.5" />
      );
    }
    return circles;
  }

  // then generate a line in between each circles:
  function generateLines() {
    let lines = [];
    let padding = 4;
    for (let i = 0; i < nCircles - 1; i++) {
      let x = circleRadius + i * (circleRadius * 2 + circleSpacing) + padding;
      let y = circleRadius;
      let x2 = circleRadius + (i + 1) * (circleRadius * 2 + circleSpacing) - padding;
      let y2 = circleRadius;
      lines.push(
        <line
          x1={x}
          y1={y}
          x2={x2}
          y2={y2}
          stroke="#000000"
          stroke-width="1"
          stroke-linecap="round"
        />
      );
    }
    return lines;
  }

  function generateHandles() {
    // one handle per circle
    let handles = [];
    let x_center = circleRadius + ((nCircles - 1) * (circleRadius * 2 + circleSpacing)) / 2;

    for (let i = 0; i < nCircles; i++) {
      let x = circleRadius + i * (circleRadius * 2 + circleSpacing);
      handles.push(
        <Handle type="source" position={Position.Bottom} id={i+1} style={{ bottom: 0, left: x }} />
      );
    }
    return handles;
  }

  let circles = generateCircles();
  let lines = generateLines();
  let handles = generateHandles();
  return (
    <div className="input-node">
      <svg
        width={nCircles * (circleRadius * 2 + circleSpacing) + circleRadius * 2}
        height={circleRadius * 2}
        viewBox={`0 0 ${nCircles * (circleRadius * 2 + circleSpacing) + circleRadius * 2} ${
          circleRadius * 2
        }`}
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {circles}
        {lines}
      </svg>
      {Util.displayCopyNumber(props.data)}
      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
      {handles}
    </div>
  );
}
export default AGGNode;
