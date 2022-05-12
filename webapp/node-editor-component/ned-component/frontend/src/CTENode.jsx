import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

const zeroPad = (num, places) => String(num).padStart(places, "0")
function CTENode(props) {
  console.log(props.data)
  return (
    <div className="cte-node">
      <svg version="1.1" viewBox="1220.0052 -491.25 51.036534 33.458246" width="50" height="30">
        <path
          d="M 1233.2951 -490.0185 L 1221.0359 -476.5394 C 1219.995 -475.395 1219.995 -473.6468 1221.0359 -472.50236 L 1233.2951 -459.02324 C 1233.8637 -458.3981 1234.6695 -458.04175 1235.5145 -458.04175 L 1267.7918 -458.04175 C 1269.4486 -458.04175 1270.7918 -459.3849 1270.7918 -461.04175 L 1270.7918 -488 C 1270.7918 -489.65685 1269.4486 -491 1267.7918 -491 L 1235.5145 -491 C 1234.6695 -491 1233.8637 -490.64364 1233.2951 -490.0185 Z"
          stroke="black"
          strokeLinecap="round"
          fill="none"
          strokeLinejoin="round"
          strokeWidth=".5"
        />
        <text transform="translate(1238 -480)" fill="black">
          <tspan font-size="8" font-weight="300" fill="black" x="0" y="8">
            BIAS {zeroPad(props.data.gdf_output, 2)}
          </tspan>
        </text>
      </svg>
      <Handle type="source" position={Position.Left} style={{ left: 2, top: 15 }} />
    </div>
  )
}

export default CTENode
