import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"
import ReactTooltip from "react-tooltip"

function DNANode({ data }: any) {
	const content = data.content.map((c:string) => <li>{c}</li>);
  return (
    <div data-tip data-for={"node_" + data.id} className="dna-node">
        <ul>{content}</ul>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default DNANode
