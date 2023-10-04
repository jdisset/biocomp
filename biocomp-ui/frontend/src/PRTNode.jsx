import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "reactflow"

function PRTNode(props) {
  return (
    <div className="prt-node-border-wrap">
    <div className="prt-node">
		<ul> <li>{props.data.content[0]} </li> </ul>
      <Handle type="target" position={Position.Top} />
		<div className="prt-name"> PRT </div>
    </div>
    </div>
  )
}

export default PRTNode
