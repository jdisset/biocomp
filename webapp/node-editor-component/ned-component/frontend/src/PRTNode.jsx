import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

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


      //<ReactTooltip id={"node_" + data.id} aria-haspopup="true">
        //<ul>{content}</ul>
      //</ReactTooltip>
