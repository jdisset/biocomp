import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

function RNANode(props) {
  const content = props.data.content.map((c) => <li> {c} </li>)
  return (
    <div className="rna-node-border-wrap">
    <div className="rna-node">
      <ul> {content} </ul>
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />
		<div className="rna-name"> RNA </div>
    </div>
    </div>
  )
}

export default RNANode
