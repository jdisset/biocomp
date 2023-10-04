import React, { ReactNode, useCallback } from "react";
import { Handle, Position } from "reactflow";

const zeroPad = (num, places) => String(num).padStart(places, "0");
function TLNode(props) {
  return (
    <div className="translation-node">
      <svg version="1.1" viewBox="1108.75 -290.25 44.5 72" width="30" height="50">
        <path
          d="M 1151.1743 -273.42278 L 1134.1743 -287.39166 C 1132.3296 -288.9075 1129.6704 -288.9075 1127.8257 -287.39166 L 1110.8257 -273.42278 C 1109.6698 -272.47303 1109 -271.05565 1109 -269.55966 L 1109 -243.24034 C 1109 -241.74435 1109.6698 -240.32696 1110.8257 -239.37722 L 1127.8257 -225.40834 C 1129.6704 -223.8925 1132.3296 -223.8925 1134.1743 -225.40834 L 1151.1743 -239.37722 C 1152.3302 -240.32696 1153 -241.74435 1153 -243.24034 L 1153 -269.55966 C 1153 -271.05565 1152.3302 -272.47303 1151.1743 -273.42278 Z"
          stroke="black"
          strokeWidth=".75"
          fill="#aaccc0"
        />
        <path
          d="M 1151.1743 -268.42278 L 1134.1743 -282.39166 C 1132.3296 -283.9075 1129.6704 -283.9075 1127.8257 -282.39166 L 1110.8257 -268.42278 C 1109.6698 -267.47303 1109 -266.05565 1109 -264.55966 L 1109 -238.24034 C 1109 -236.74435 1109.6698 -235.32696 1110.8257 -234.37722 L 1127.8257 -220.40834 C 1129.6704 -218.8925 1132.3296 -218.8925 1134.1743 -220.40834 L 1151.1743 -234.37722 C 1152.3302 -235.32696 1153 -236.74435 1153 -238.24034 L 1153 -264.55966 C 1153 -266.05565 1152.3302 -267.47303 1151.1743 -268.42278 Z"
          stroke="black"
          strokeWidth=".75"
          fill="none"
        />
        <text transform="translate(1114 -262)" fill="black" x="10" y="11">
          {zeroPad(props.data.id, 2)}
        </text>
      </svg>
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} style={{ bottom: 0 }} />
    </div>
  );
}

export default TLNode;
