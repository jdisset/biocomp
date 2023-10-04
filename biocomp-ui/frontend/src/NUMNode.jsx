import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "reactflow";
import { theme } from "./shapes.jsx";
import Util from "./util.jsx";
function NUMNode(props) {
  return (
    <div className="numeric-node">
      {props.data.extra && props.data.extra.value && (
        <div className="numeric-node-value">{props.data.extra.value}</div>
      )}

      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
    </div>
  );
}
export default NUMNode;
