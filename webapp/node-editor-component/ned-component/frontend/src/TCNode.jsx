import React, { ReactNode, useCallback } from "react"
import { Handle, Position } from "react-flow-renderer"
const zeroPad = (num, places) => String(num).padStart(places, "0")

function TCNode(props) {
  return (
    <div className="transcription-node">
      <svg version="1.1" viewBox="1108.75 -412.80825 44.5 72.7" width="30" height="50">
        <path
          d="M 1151.3091 -394.6286 L 1134.3091 -409.63683 C 1132.4187 -411.3058 1129.5813 -411.3058 1127.6909 -409.63683 L 1110.6909 -394.6286 C 1109.6158 -393.67947 1109 -392.3144 1109 -390.8803 L 1109 -362.0362 C 1109 -360.6021 1109.6158 -359.23703 1110.6909 -358.2879 L 1127.6909 -343.27968 C 1129.5813 -341.61073 1132.4187 -341.61073 1134.3091 -343.27968 L 1151.3091 -358.2879 C 1152.3842 -359.23703 1153 -360.6021 1153 -362.0362 L 1153 -390.8803 C 1153 -392.3144 1152.3842 -393.67947 1151.3091 -394.6286 Z"
          stroke="black"
          fill="none"
          strokeWidth=".75"
        />
        <text transform="translate(1114 -384)" fill="black" x="10" y="11">
			{zeroPad(props.data.id,2)}
        </text>
      </svg>
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default TCNode
