import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "react-flow-renderer";
import { TShape, theme } from "./shapes.jsx";
import Util from "./util.jsx";

function TLNode(props) {
	// name should be TL + id
  name = "TL " + props.id;
  return (
    <div className="translation-node">
      <TShape txt={name} topcolor={theme.RNAcolor} bottomcolor={theme.PRTcolor} />
      <Handle type="target" position={Position.Top} style={{ top: 0 }} />
      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
    </div>
  );
}

export default TLNode;
