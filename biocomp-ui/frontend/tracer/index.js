import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import SliderAxis from "./SliderAxis"; // import the SliderAxis component
import { layoutData, pointData } from "./data"; // import your data
import * as d3 from "d3";
import "./style.css";

const AXIS_OFFSET = { x: 10, y: 53 };
const AXIS_WIDTH = 200;

function App() {
  const sliderRefs = layoutData.map((row, rowIndex) => {
    return row.map((item, columnIndex) => {
      return React.createRef();
    });
  });

  // create ranges in a 2D list
  const init_ranges = layoutData.map((row, rowIndex) => {
    return row.map((item, columnIndex) => {
      return [0, 1];
    });
  });

  const [ranges, setRanges] = useState(init_ranges);

  const setSliderRange = (rowIndex, columnIndex, range) => {
    setRanges((prev) => {
      const newRanges = [...prev];
      newRanges[rowIndex][columnIndex] = range;
      return newRanges;
    });
  };

  const svgRef = useRef();

  const buildSelected = () => {
    let selected = [];
    for (let rowIndex in pointData) {
      for (let columnIndex in pointData[rowIndex]) {
        const minVal = ranges[rowIndex][columnIndex][0];
        const maxVal = ranges[rowIndex][columnIndex][1];
        let pointIsSelected = pointData[rowIndex][columnIndex][0].map(
          (d) => d >= minVal && d <= maxVal
        );
        // a point is selected if it is selected by all slider
        if (selected.length === 0) {
          selected = pointIsSelected;
        }
        selected = selected.map((d, i) => d && pointIsSelected[i]);
      }
    }
    return selected;
  };

  useEffect(() => {
    if (!svgRef.current) {
      return;
    }
    const svg = d3.select(svgRef.current);
    // clear svg
    svg.selectAll("*").remove();

    // display points for every slider
    const selected = buildSelected();

    // create position info with the same structure as pointData
    let positions = pointData.map((row) => {
      return row.map((item) => {
        return null;
      });
    });


    for (let rowIndex in pointData) {
      for (let columnIndex in pointData[rowIndex]) {
        const ref = sliderRefs[rowIndex][columnIndex];
        if (!ref.current) {
          continue;
        }
        let rect = ref.current.getBoundingClientRect();
        rect = {
          x: rect.x + window.scrollX,
          y: rect.y + window.scrollY,
        };

        const minVal = ranges[rowIndex][columnIndex][0];
        const maxVal = ranges[rowIndex][columnIndex][1];
        let points = pointData[rowIndex][columnIndex][0];
        let pos = points.map((d) => {
          return {
            x: rect.x + AXIS_OFFSET.x + d * AXIS_WIDTH,
            y: rect.y + AXIS_OFFSET.y,
          };
        });
        positions[rowIndex][columnIndex] = pos;


        // draw points
        svg
          .selectAll(`.point`)
          .data(pos)
          .enter()
          .append("circle")
          .attr("cx", (d) => d.x)
          .attr("cy", (d) => d.y)
          .attr("r", (d, i) => (selected[i] ? 4 : 1))
          .attr("fill", (d, i) => (selected[i] ? "#00B077" : "#999999"))
          .raise();
      }
    }

    // now draw lines between points. for that we have to use tthe layoutData, which
    // for each [rowIndex][columnIndex] contains a output_to object
    // that contains the rowIndex and columnIndex of the next slider
    for (let r=0; r<layoutData.length; r++) {
      for (let c=0; c<layoutData[r].length; c++) {
        const row = layoutData[r];
        const item = row[c];
        if (item.output_to.length === 0) {
          continue;
        }
        const nextRow = item.output_to[0][0];
        const nextColumn = item.output_to[0][1];
        if (nextRow === undefined || nextColumn === undefined) {
          continue;
        }
        const pos1 = positions[r][c];
        const pos2 = positions[nextRow][nextColumn];
        for (let i=0; i<pos1.length; i++) {
          svg
            .append("line")
            .attr("x1", pos1[i].x)
            .attr("y1", pos1[i].y)
            .attr("x2", pos2[i].x)
            .attr("y2", pos2[i].y)
            .attr("stroke", selected[i] ? "#00B077" : "#AAA")
            .attr("stroke-width", selected[i] ? 2 : 0.5)
            .lower();
        }
      }
    }


  }, [svgRef, ranges, sliderRefs, layoutData, pointData]);

  return (
    <div>
      <svg className="traces" ref={svgRef} />
      {layoutData.map((row, rowIndex) => (
        <div key={`row-${rowIndex}`} style={{ display: "flex", flexDirection: "row" }}>
          {row.map((item, columnIndex) => {
            return (
              <SliderAxis
                key={`slider-${rowIndex}-${columnIndex}`}
                sliderData={item}
                points={pointData[rowIndex][columnIndex]}
                ref={sliderRefs[rowIndex][columnIndex]}
                setSliderRange={(range) => setSliderRange(rowIndex, columnIndex, range)}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
