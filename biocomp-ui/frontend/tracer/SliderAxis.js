import React, { useEffect, useRef, useState, forwardRef } from "react";
import * as d3 from "d3";
import { Range, getTrackBackground } from "react-range";

const SliderAxis = forwardRef(({ sliderData, points, setSliderRange }, ref) => {
  const [values, setValues] = useState([0, 1]);

  /*  const svgRef = useRef();*/

  const svgRef = useRef();
  useEffect(() => {
    if (!svgRef.current) {
      return;
    }
    const svg = d3.select(svgRef.current);
    // Initialize the scales
    const xScale = d3.scaleLinear().domain([0, 1]).range([10, 210]); // set the range depending on your desired width
    // Create the axis
    const tickFormat = d3.format(".1f");
    const axis = d3.axisBottom(xScale).ticks(3).tickFormat(tickFormat).tickSize(5);
    // Add the axis to the svg
    svg.append("g").call(axis);
    svg.selectAll(".tick text").attr("font-size", "10").attr("fill", "#999999");
    svg.selectAll(".tick line").attr("stroke", "#999999");
    svg.selectAll(".domain").attr("stroke", "#999999");
    svg.attr("viewBox", "0 -5 220 30");
  }, [svgRef]);


  const STEP = 0.005;
  const MIN = 0;
  const MAX = 1;

  // Copy of TwoThumbs with `draggableTrack` prop added
  return (
    <div className="slideraxis" ref={ref}>
      {sliderData.name}
      <svg ref={svgRef} className="axis"></svg>
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
            style={{
              ...props.style,
              height: "36px",
              display: "flex",
              width: "100%",
            }}
          >
            <div
              ref={props.ref}
              style={{
                height: "8px",
                width: "100%",
                borderRadius: "4px",
                background: getTrackBackground({
                  values,
                  colors: ["#00000020", "#00B07725", "#00000020"],
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
