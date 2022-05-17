import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"
import Util from "./util.jsx"
function hasCopyNunmber(data) {
  return data.parameters && data.parameters.copy_number >= 0
}

function displayCopyNumber(data) {
  if (hasCopyNunmber(data)) {
    let cn = data.parameters.copy_number
    let col = Util.cmap(cn)
    return (
      <div className="copy_number">
        <svg version="1.1" viewBox="0 0 37 37" width="35" height="35">
          <circle cx="18" cy="18" r="15" stroke={col} strokeWidth={0.5 + cn*3.0} fill="white" />
          <text font-size="10" fontWeight="300" transform="translate(11 22)" fill="black">
            {cn.toFixed(1)}
          </text>
        </svg>
      </div>
    )
  }
}

const zeroPad = (num, places) => String(num).padStart(places, "0")
function INNode(props) {
  console.log(props)
  return (
    <div className="input-node">
      <svg version="1.1" viewBox="1108.75 -525.25 44.5 62.58349" width="44.5" height="62.58349">
        <path
          d="M 1110.83 -479.4519 L 1127.83 -465.51524 C 1129.6732 -464.0042 1132.3268 -464.0042 1134.17 -465.51524 L 1151.17 -479.4519 C 1152.3284 -480.4016 1153 -481.82064 1153 -483.3186 L 1153 -520 C 1153 -522.7614 1150.7614 -525 1148 -525 L 1114 -525 C 1111.2386 -525 1109 -522.7614 1109 -520 L 1109 -483.3186 C 1109 -481.82064 1109.6716 -480.4016 1110.83 -479.4519 Z"
          stroke="black"
          strokeWidth=".5"
          fill="white"
        />
        <text transform="translate(1117 -510)" fill="black" x="10" y="11">
          <tspan fontSize="10" fontWeight="300" fill="black" x="0" y="9">
            INPUT
          </tspan>
          <tspan fontSize="12" fontWeight="300" fill="black" x="7" y="25">
            {zeroPad(props.data.gdf_output, 2)}
          </tspan>
        </text>
      </svg>
      {displayCopyNumber(props.data)}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ bottom: hasCopyNunmber(props.data) ? 40 : 10 }}
      />
    </div>
  )
}

export default INNode
