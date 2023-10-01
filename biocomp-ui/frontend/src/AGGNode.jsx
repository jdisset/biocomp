import React, { ReactNode, useCallback, useState, useEffect } from "react";
import { Handle, Position } from "react-flow-renderer";
import { theme } from "./shapes.jsx";
import Util from "./util.jsx";
import ParamInput from "./ParamInput.jsx";

function AGGNode(props) {
  let nCircles = props.data.output_to.length;
  let circleRadius = 15;
  let circleSpacing = 8;

  function generateCircles(ratios = null) {
    let circles = [];
    for (let i = 0; i < nCircles; i++) {
      let x = circleRadius + i * (circleRadius * 2 + circleSpacing);
      let y = circleRadius;
      circles.push(
        <circle
          cx={x}
          cy={y}
          r={circleRadius - 0.5}
          stroke="black"
          strokeWidth="0.5"
          key={"agg_circ_" + i}
          onMouseDown={(e) => handleCircleClick(e, i, x - 1, y - 1)}
        />,
      );

      if (ratios) {
        console.assert(ratios.length == nCircles);
        circles.push(
          <text
            key={"agg_text_" + i}
            x={x}
            y={y + 3}
            textAnchor="middle"
            fill="black"
            fontSize="8px"
            letterSpacing="0"
            onMouseDown={(e) => handleCircleClick(e, i, x - 1, y - 1)}
          >
            {ratios[i].toFixed(2)}
          </text>,
        );
      }
    }
    return circles;
  }

  // then generate a line in between each circles:
  function generateLines() {
    let lines = [];
    let padding = 14;
    for (let i = 0; i < nCircles - 1; i++) {
      let x = circleRadius + i * (circleRadius * 2 + circleSpacing) + padding;
      let y = circleRadius;
      let x2 = circleRadius + (i + 1) * (circleRadius * 2 + circleSpacing) - padding;
      let y2 = circleRadius;
      lines.push(
        <line
          key={"agg_line_" + i}
          x1={x}
          y1={y}
          x2={x2}
          y2={y2}
          stroke="#000000"
          strokeWidth="1"
          strokeLinecap="round"
          className="drag-handle"
        />,
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
        <Handle
          key={"agg_handle_" + i}
          type="source"
          position={Position.Bottom}
          id={i + 1}
          style={{ bottom: 0, left: x }}
        />,
      );
    }
    return handles;
  }

  const lines = generateLines();
  const handles = generateHandles();

  const [ratios, setRatios] = useState(props.data.extra.ratios);
  const [circles, setCircles] = useState(generateCircles(ratios));

  useEffect(() => {
    if ("tunable" in props.data) {
      for (const [path, i, name, value] of props.data.tunable) {
        if (name == "ratios") {
          setRatios(value);
        }
      }
    } else if (props.data.extra && props.data.extra.ratios) {
      setRatios(props.data.extra.ratios);
    }
  }, [props.data]);

  useEffect(() => {
    setCircles(generateCircles(ratios));
  }, [ratios]);

  useEffect(() => {
    if ("tunable" in props.data) {
      let new_tunable = [];
      for (const [path, i, name, value] of props.data.tunable) {
        if (name == "ratios") {
          new_tunable.push([path, i, name, ratios]);
        } else {
          new_tunable.push([path, i, name, value]);
        }
      }
      if (props.data.updateMyParams) {
        props.data.updateMyParams(new_tunable);
      }
    }
  }, [ratios]);

  function updateRatio(index, value) {
    let new_ratios = [...ratios];
    new_ratios[index] = parseFloat(value);
    setRatios(new_ratios);
  }

  const [paramProps, setParamProps] = useState({
    param_position: { x: 0, y: 0 },
    mouse_position: { x: 0, y: 0 },
    display: false,
    value: 0,
    onChange: null,
  });

  function handleCircleClick(event, index, x, y) {
    if (ratios) {
      let value = ratios[index];
      const callback = (value) => {
        updateRatio(index, value);
      };
      setParamProps({
        param_position: { x: x, y: y },
        mouse_position: { x: event.clientX, y: event.clientY },
        display: true,
        value: value,
        onChange: callback,
      });
    }
  }

  const total_width = nCircles * (circleRadius * 2 + circleSpacing) - circleSpacing;
  return (
    <>
      <ParamInput
        isDragging={paramProps.display}
        startMousePosition={paramProps.mouse_position}
        startValue={paramProps.value}
        objPosition={paramProps.param_position}
        onChange={paramProps.onChange}
        onClose={() => setParamProps({ ...paramProps, display: false })}
      />

      <div className="input-node">
        <svg
          width={total_width}
          height={circleRadius * 2}
          viewBox={`0 0 ${total_width} ${circleRadius * 2}`}
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {circles}
          {lines}
          <text
            x={total_width / 2}
            y={circleRadius * 2}
            textAnchor="middle"
            fill="black"
            fontSize="8px"
            letterSpacing="0"
          >
            {props.data.id}
          </text>
        </svg>
        {handles}
        <Handle type="target" position={Position.Top} style={{ top: 0, left: total_width / 2 }} />
      </div>
    </>
  );
}
export default AGGNode;
