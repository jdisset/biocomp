import React, { ReactNode, useCallback, useState, useEffect } from "react";
import ReactFlow, { getBezierPath, useReactFlow } from 'reactflow';
import IndividualParamInput from "./IndividualParamInput.jsx";

const baseHeight = 20;
const width = 55;

export default function CustomEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data = {},
  markerEnd,
}) {
  const [edgePath, edgeCenterX, edgeCenterY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const reactFlowInstance = useReactFlow();

  const initialMouseDownInfo = { event: null, x: -1, y: -1 };
  const [mouseDownInfo, setMouseDownInfo] = useState(initialMouseDownInfo);

  let itemsAsText = "";
  let pinput = "";


  if (data.srccdg && (data.srcdata.type == "transcription" || data.srcdata.type == "translation")) {

    const content = data.srccdg.content;
    const params = data.srccdg.params ? data.srccdg.params : {};
    // params is a dictionary of the parameters of the edge




    const rateName = data.srcdata.type == "transcription" ? "tc_rate" : "tl_rate";

    // TODO: let the relevant nodes set this edge's tunable param callback
    //pinput = (
      //<IndividualParamInput
        //mouseDownEvent={mouseDownInfo.event}
        //objpos={{ x: mouseDownInfo.x, y: mouseDownInfo.y }}
        //clearMouseDownEvent={() => setMouseDownInfo(initialMouseDownInfo)}
        //pname={rateName}
        //subname={0}
        //tunableData={props.data.tunable}
        //updateParams={props.data.updateMyParams}
        //pvalue={value}
        //setPValue={setValue}
      ///>
    //)

    //console.log(data)

    itemsAsText = (
      <>
        <text
          x={edgeCenterX}
          y={edgeCenterY}
          textAnchor="middle"
          alignmentBaseline="central"
          stroke="white"
          strokeWidth="10"
          className="edgecontent-text"
        >
          {content.join(" + ") +
            "   " +
            (Object.keys(params).length > 0 ? JSON.stringify(params) : "")}
        </text>

        <text
          x={edgeCenterX}
          y={edgeCenterY}
          fontSize="10"
          textAnchor="middle"
          alignmentBaseline="central"
          className="edgecontent-text"
        >
          {content.join(" + ") +
            "   " +
            (Object.keys(params).length > 0 ? JSON.stringify(params) : "")}
        </text>
      </>
    );
  }

  return (
    <>
      {pinput}
      <path
        id={id}
        style={style}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      {itemsAsText}
    </>
  );
}
