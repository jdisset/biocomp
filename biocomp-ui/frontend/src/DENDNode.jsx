import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "reactflow"

function DENDNode(props) {
	// let's make a cross symbol with svg
  return (
    <div className="out-node">
		<svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
			<line x1="0" y1="0" x2="20" y2="20" stroke="black" strokeWidth="1"/>
			<line x1="20" y1="0" x2="0" y2="20" stroke="black" strokeWidth="1"/>
      </svg>

      <Handle type="target" position={Position.Top} style={{ top: 8 }} />
    </div>
  )
}

export default DENDNode
