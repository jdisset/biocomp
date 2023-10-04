import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "reactflow";
import Util from "./util.jsx";

function TLNode(props) {
  return (
    <div className="inverse-node">
      {props.data.type + " " + props.id}
      <Handle type="target" position={Position.Top} style={{ top: 0 }} />
      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
    </div>
  );
}

export default TLNode;
