import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

function SEQNode(props) {
	const shortName = (props.data.type === 'sequestron_ERN') ? 'ERN' : 'RCB';
  return (
    <div className="seq-node">
      <Handle type="target" position={Position.Top} id="0" style={{ left: 60 - 25 }} />
      <Handle type="target" position={Position.Top} id="1" style={{ left: 60 + 25 }} />
      <Handle type="source" position={Position.Bottom} />
      <div className="seq-name"> {shortName} </div>
    </div>
  )
}

export default SEQNode
