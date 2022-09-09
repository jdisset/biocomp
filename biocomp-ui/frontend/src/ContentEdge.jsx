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
  if (data.srccdg && (data.srcdata.type == "transcription" || data.srcdata.type == "translation")) {
    const content = data.srccdg.content;
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
      {foreignObject}
    </>
  );
}
