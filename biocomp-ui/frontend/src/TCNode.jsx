import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "react-flow-renderer";
import { TShape, theme } from "./shapes.jsx";
import Util from "./util.jsx";

function TCNode(props) {
  name = "TC " + props.id;
  return (
    <div className="transcription-node">
      <TShape txt={name} topcolor={theme.DNAcolor} bottomcolor={theme.RNAcolor} />
      <Handle type="target" position={Position.Top} style={{ top: 0 }} />
      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
    </div>
  );
}

export default TCNode;
