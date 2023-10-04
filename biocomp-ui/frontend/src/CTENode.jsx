import React, { ReactNode, useCallback, useState, useEffect } from "react";
import { Handle, Position } from "reactflow";
import Util from "./util.jsx";
import { theme } from "./shapes.jsx";
import IndividualParamInput from "./IndividualParamInput.jsx";

function CTENode(props) {
  const initialMouseDownInfo = { event: null, x: -1, y: -1 };
  const [mouseDownInfo, setMouseDownInfo] = useState(initialMouseDownInfo);
  const [value, setValue] = useState([1]);


  return (
    <>

      <IndividualParamInput
        mouseDownEvent={mouseDownInfo.event}
        objpos={{ x: mouseDownInfo.x, y: mouseDownInfo.y }}
        clearMouseDownEvent={() => setMouseDownInfo(initialMouseDownInfo)}
        pname="value"
        subname={0}
        tunableData={props.data.tunable}
        updateParams={props.data.updateMyParams}
        pvalue={value}
        setPValue={setValue}
      />

      <div className="cte-node">
        <svg version="1.1" viewBox="600 -1300 75 33" width="75" height="33">
          <path
            d="M 620.3428 -1299.5 C 620.3428 -1299.5 620.3428 -1299.5 620.3428 -1299.5 Z M 620.3428 -1299.5 L 616.5428 -1299.5 C 615.0948 -1299.5 613.7179 -1298.8723 612.7683 -1297.7792 L 603.2119 -1286.7792 C 601.5779 -1284.8984 601.5779 -1282.1016 603.2119 -1280.2208 L 612.7683 -1269.2208 C 613.7179 -1268.1277 615.0948 -1267.5 616.5428 -1267.5 L 620.3428 -1267.5 L 635.23496 -1267.5 L 635.85524 -1267.5 L 639.03496 -1267.5 L 639.65524 -1267.5 L 648.03496 -1267.5 L 651.83496 -1267.5 C 653.4918 -1267.5 654.83496 -1268.8431 654.83496 -1270.5 L 654.83496 -1296.5 C 654.83496 -1298.1569 653.4918 -1299.5 651.83496 -1299.5 L 648.03496 -1299.5 L 639.65524 -1299.5 L 639.03496 -1299.5 L 635.85524 -1299.5 L 635.23496 -1299.5 Z"
            fill="#ffb703"
          />
          <path
            d="M 620.3428 -1299.5 C 620.3428 -1299.5 620.3428 -1299.5 620.3428 -1299.5 Z M 620.3428 -1299.5 L 616.5428 -1299.5 C 615.0948 -1299.5 613.7179 -1298.8723 612.7683 -1297.7792 L 603.2119 -1286.7792 C 601.5779 -1284.8984 601.5779 -1282.1016 603.2119 -1280.2208 L 612.7683 -1269.2208 C 613.7179 -1268.1277 615.0948 -1267.5 616.5428 -1267.5 L 620.3428 -1267.5 L 635.23496 -1267.5 L 635.85524 -1267.5 L 639.03496 -1267.5 L 639.65524 -1267.5 L 648.03496 -1267.5 L 651.83496 -1267.5 C 653.4918 -1267.5 654.83496 -1268.8431 654.83496 -1270.5 L 654.83496 -1296.5 C 654.83496 -1298.1569 653.4918 -1299.5 651.83496 -1299.5 L 648.03496 -1299.5 L 639.65524 -1299.5 L 639.03496 -1299.5 L 635.85524 -1299.5 L 635.23496 -1299.5 Z"
            stroke="black"
            fill={theme.DNAcolor}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth=".5"
          />
          <path
            d="M 639.51714 -1299.2347 L 638.8994 -1299.2347 L 620.28344 -1299.2347 C 618.84135 -1299.2347 617.4701 -1298.6174 616.5243 -1297.5424 L 607.0069 -1286.7248 C 605.3796 -1284.8752 605.3796 -1282.1248 607.0069 -1280.2752 L 616.5243 -1269.4576 C 617.4701 -1268.3826 618.84135 -1267.7653 620.28344 -1267.7653 L 638.8994 -1267.7653 L 639.51714 -1267.7653 L 651.6472 -1267.7653 C 653.2973 -1267.7653 654.63496 -1269.0862 654.63496 -1270.7155 L 654.63496 -1296.2845 C 654.63496 -1297.9138 653.2973 -1299.2347 651.6472 -1299.2347 Z"
            fill="white"
          />
          <text transform="translate(620.3447 -1290.537)" fill="#322f30">
            <tspan fontFamily="Roboto" fontSize="11" fontWeight="400" fill="#322f30" x="0" y="10">
              Bias
            </tspan>
          </text>


        </svg>
        {Util.displayCopyNumber(value[0], 'black', (e) => setMouseDownInfo({ event: e, x: 0, y: 0 }))}

        <Handle type="source" position={Position.Left} style={{ left: 2, top: 15 }} />
      </div>
    </>
  );
}

export default CTENode;

