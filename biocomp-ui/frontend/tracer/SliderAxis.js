import React, { useEffect, useRef, useState, forwardRef } from "react";
import * as d3 from "d3";
import { COLORS } from "./constants";

import Draggable, { DraggableCore } from "react-draggable";

const RangeSlider = ({ values, step, min, max, onChange, width, scale }) => {
  const [value1, setValue1] = useState(values[0]);
  const [value2, setValue2] = useState(values[1]);

  const scaler = (max - min) / width;

  const onDrag = (e, ui, handleType) => {
    let newValue1 = value1;
    let newValue2 = value2;
    let dx = ui.deltaX * scaler;

    if (handleType === "min" || handleType === "range") newValue1 += dx;
    if (handleType === "max" || handleType === "range") newValue2 += dx;

    newValue1 = Math.max(min, newValue1);
    newValue1 = Math.min(newValue1, value2 - step);
    setValue1(newValue1);
    setValue2(newValue2);
    onChange([newValue1, newValue2]);
  };

  const thumb1Pos = (value1 - min) / scaler;
  const thumb2Pos = (value2 - min) / scaler;
  const gridstep = step / scaler;

  return (
    <div className="range-slider">
      <div className="slider-track" style={{ width: width }}></div>
      <Draggable
        onDrag={(e, ui) => onDrag(e, ui, "range")}
        axis="x"
        grid={[gridstep, 0]}
        bounds={{ left: 0, right: width - (thumb2Pos - thumb1Pos) }}
        position={{ x: thumb1Pos, y: 0 }}
        scale={scale}
      >
        <div className="slider-thumb range-thumb" style={{ width: thumb2Pos - thumb1Pos }}></div>
      </Draggable>
      <Draggable
        onDrag={(e, ui) => onDrag(e, ui, "min")}
        axis="x"
        grid={[gridstep, 0]}
        bounds={{ left: 0, right: thumb2Pos }}
        position={{ x: thumb1Pos, y: 0 }}
        scale={scale}
      >
        <div className="slider-thumb min-thumb"></div>
      </Draggable>

      <Draggable
        onDrag={(e, ui) => onDrag(e, ui, "max")}
        axis="x"
        bounds={{ left: thumb1Pos, right: width }}
        grid={[gridstep, 0]}
        position={{ x: thumb2Pos, y: 0 }}
        scale={scale}
      >
        <div className="slider-thumb max-thumb"></div>
      </Draggable>
    </div>
  );
};

const SliderAxis = forwardRef(({ sliderData, min, max, setSliderRange, style, scale }, ref) => {
  const STEP = 0.001;
  const [values, setValues] = useState([min, max]);

  const width = 200;

  const truncateStr = (str, max) => {
    if (str === "sequestron_ERN") return "ERN";
    if (str.length > max) return str.substring(0, max - 3) + "...";
    return str;
  };

  useEffect(() => {
    setSliderRange(values);
  }, [values]);

  // Copy of TwoThumbs with `draggableTrack` prop added
  return (
    <div className="slideraxis" ref={ref} style={style}>
      <div className={`slider-labels ${sliderData.type}`}>
        <div className="type" style={{ background: COLORS[sliderData.type] }}>
          <span> {sliderData.node_id}{sliderData.n_outputs > 1 ? `.${sliderData.slot}` : ""} </span>
          {truncateStr(sliderData.type, 14)}
        </div>
        {sliderData.info && <div className="info">{sliderData.info}</div>}
      </div>
      <RangeSlider
        draggableTrack
        values={values}
        step={STEP}
        min={min}
        max={max}
        width={width}
        scale={scale}
        onChange={(values) => {
          setValues(values);
        }}
      />
    </div>
  );
});

export default SliderAxis;
