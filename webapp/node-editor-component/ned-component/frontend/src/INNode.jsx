import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"

const zeroPad = (num, places) => String(num).padStart(places, "0")
function INNode(props) {
  return (
    <div className="input-node">
      <svg version="1.1" viewBox="1108.75 -525.25 44.5 62.58349" width="44.5" height="62.58349">
        <path
          d="M 1110.83 -479.4519 L 1127.83 -465.51524 C 1129.6732 -464.0042 1132.3268 -464.0042 1134.17 -465.51524 L 1151.17 -479.4519 C 1152.3284 -480.4016 1153 -481.82064 1153 -483.3186 L 1153 -520 C 1153 -522.7614 1150.7614 -525 1148 -525 L 1114 -525 C 1111.2386 -525 1109 -522.7614 1109 -520 L 1109 -483.3186 C 1109 -481.82064 1109.6716 -480.4016 1110.83 -479.4519 Z"
          stroke="black"
          strokeWidth=".5"
          fill="none"
        />
        <text transform="translate(1114 -509.4791)" fill="black" x="10" y="11">
			{zeroPad(props.data.id,2)}
        </text>
      </svg>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default INNode
