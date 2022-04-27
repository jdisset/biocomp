import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

function PRTNode({ data }: any) {
	const content = data.content.map((c:string) => <em>{c}</em>);
  return (
    <div className="text-updater-node">
        {content}
      <Handle type="target" position={Position.Top} />
    </div>
  )
}

export default PRTNode


      //<ReactTooltip id={"node_" + data.id} aria-haspopup="true">
        //<ul>{content}</ul>
      //</ReactTooltip>
