import React, { ReactNode, useCallback } from "react";
import { ERNSeqShape, RCBSeqShape, theme } from "./shapes.jsx";
import { Handle, Position } from "reactflow";

const zeroPad = (num, places) => String(num).padStart(places, "0");
const getShape = (name, props) => {
  if (name.normalize() === "ERN ".normalize()) {
    return (
      <ERNSeqShape
        txt={name + zeroPad(props.id, 2)}
        leftcolor={theme.PRTcolor}
        rightcolor={theme.RNAcolor}
        outcolor={theme.RNAcolor}
      />
    );
  } else {
    return (
      <RCBSeqShape
        txt={name + zeroPad(props.id, 2)}
        leftcolor={theme.PRTcolor}
        rightcolor={theme.PRTcolor}
        outcolor={theme.DNAcolor}
      />
    );
  }
};

function SEQNode(props) {
  const shortName = props.data.type === "sequestron_ERN" ? "ERN " : "RCB ";
  return (
    <div className="seq-node">
      {getShape(shortName, props)}
      <Handle type="target" position={Position.Top} id="0" style={{ left: 60 - 22 }} />
      <Handle type="target" position={Position.Top} id="1" style={{ left: 60 + 36 }} />
      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
    </div>
  );
}

export default SEQNode;
