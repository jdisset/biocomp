import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "reactflow"

function OUTNode(props) {
  return (
    <div className="out-node">
      <svg
        version="1.1"
        viewBox="1429.8046 -347.12437 13.410718 7.455359"
        width="13.410718"
        height="7.455359"
      >
        <line
          x1="1442.4653"
          y1="-346.37437"
          x2="1436.51"
          y2="-340.419"
          stroke="black"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1"
        />
        <line
          x1="1436.51"
          y1="-345.5874"
          x2="1436.51"
          y2="-343.45938"
          stroke="black"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth=".5"
        />
        <line
          x1="1430.5546"
          y1="-346.37437"
          x2="1436.51"
          y2="-340.419"
          stroke="black"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1"
        />
      </svg>

      <Handle type="target" position={Position.Top} style={{ top: 8 }} />
    </div>
  )
}

export default OUTNode
