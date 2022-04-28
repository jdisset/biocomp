import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"
import ReactTooltip from "react-tooltip"
import DNAContent from "./DNAContent.jsx"

function DNANode({ data }: any) {
  return (
	<div>
      <DNAContent data={data} />
      <Handle type="source" position={Position.Bottom} />
	</div>
  )
}

export default DNANode
