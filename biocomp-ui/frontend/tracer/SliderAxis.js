import React, { useEffect, useRef, useState, forwardRef } from "react";
import * as d3 from "d3";
import { Range, getTrackBackground } from "react-range";
import {COLORS} from "./constants";

const SliderAxis = forwardRef(({ sliderData, points, setSliderRange, style }, ref) => {
  const [values, setValues] = useState([0, 1]);
  const STEP = 0.005;
  const MIN = 0;
  const MAX = 1;

  // Copy of TwoThumbs with `draggableTrack` prop added
  return (
    <div className="slideraxis" ref={ref}  style={style}>
      <div className="slider-labels">
        <div className="slider-label">{sliderData.name}</div>
      </div>

      <Range
        draggableTrack
        values={values}
        step={STEP}
        min={MIN}
        max={MAX}
        onChange={(values) => {
          setValues(values);
          setSliderRange(values);
        }}
        renderTrack={({ props, children }) => (
          <div
            className="slider"
            onMouseDown={props.onMouseDown}
            onTouchStart={props.onTouchStart}
          >
            <div
              ref={props.ref}
              style={{
                height: "1px",
                width: "100%",
                borderRadius: "4px",
                background: getTrackBackground({
                  values,
                  colors: [COLORS.out_of_range, COLORS.in_range, COLORS.out_of_range],
                  min: MIN,
                  max: MAX,
                }),
                alignSelf: "center",
              }}
            >
              {children}
            </div>
          </div>
        )}
        renderThumb={({ props, isDragged }) => (
          <div className="slider-thumb" {...props}>
            <div
              className="slider-thumb-tick"
              style={{
                backgroundColor: isDragged ? "#548BF4" : "#444",
              }}
            />
          </div>
        )}
      />
    </div>
  );
});

export default SliderAxis;
