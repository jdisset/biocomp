import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

function RNANode({ data }: any) {
  return (
    <div className="rna-node">
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default RNANode
