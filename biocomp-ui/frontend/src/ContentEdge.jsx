import React from "react";
import { getBezierPath, getEdgeCenter, getMarkerEnd } from "react-flow-renderer";

const baseHeight = 20;
const width = 55;

//const onEdgeClick = (evt, id) => {
//evt.stopPropagation();
//alert(`remove ${id}`);
//};

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
  const edgePath = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  const [edgeCenterX, edgeCenterY] = getEdgeCenter({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });

  function hasEdgeLabel(data) {
    if (data.tgtdata.parameters) {
      if (data.tgtdata.type === "transcription" || data.tgtdata.type === "translation") {
        return true;
      }
    }
    return false;
  }

  function getRate(data) {
    if (hasEdgeLabel(data)) {
      var i = parseInt(data.tgthandle);
      return data.tgtdata.parameters.tr_rates[i];
    } else return 0;
  }

  let foreignObject = "";
  let outputValue = "";
  // outputValue might have beed passed (as a string)
  if (data.outputValue !== undefined && data.outputValue !== "" && data.outputValue !== null) {
    outputValue = (
      <>
        <text
          x={edgeCenterX}
          y={edgeCenterY}
          textAnchor="start"
          alignmentBaseline="central"
          stroke="white"
          strokeWidth="10"
          className="edgecontent-text"
        >
          {parseFloat(data.outputValue).toExponential(2)}
        </text>

        <text
          x={edgeCenterX}
          y={edgeCenterY}
          fontSize="10"
          textAnchor="start"
          alignmentBaseline="central"
          className="edgecontent-text"
          fontWeight="bold"
        >
          {parseFloat(data.outputValue).toExponential(2)}
        </text>
      </>
    );
  }

  let itemsAsText = "";
  if (data.srccdg && (data.srcdata.type == "transcription" || data.srcdata.type == "translation")) {
    const content = data.srccdg.content;
	// params is a dictionary of the parameters of the edge
	var params = {};
	if (data.srccdg.params) {
		params = data.srccdg.params;
		//if (data.srcdata.type == "transcription") {
			//var params = data.srccdg.params["tc_rate"];
		//} else if (data.srcdata.type == "translation") {
			//var params = data.srccdg.params["tl_rate"];
		//}
	}

    const listItems = data.srccdg.content.map((e) => <li>{e}</li>);
    const height = baseHeight * listItems.length + 10;

    foreignObject = (
      <foreignObject
        width={width}
        height={height}
        x={edgeCenterX - width / 2}
        y={edgeCenterY - height / 2}
        className="edgecontent-foreignobject"
      >
        <body>
          <div className="edge-content">{listItems}</div>
        </body>
      </foreignObject>
    );

    // now as svg, it'll just be a text element with white background
    // and black text. We want to have a new line for each element
    // white background
    if (outputValue == "") {
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
			  {content.join(" + ") + "   " + (Object.keys(params).length > 0 ? JSON.stringify(params) : "")}
          </text>

          <text
            x={edgeCenterX}
            y={edgeCenterY}
            fontSize="10"
            textAnchor="middle"
            alignmentBaseline="central"
            className="edgecontent-text"
          >
			  {content.join(" + ") + "   " + (Object.keys(params).length > 0 ? JSON.stringify(params) : "")}
          </text>
        </>
      );
    }
  }

  return (
    <>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      {itemsAsText}
      {outputValue}
    </>
  );
}
