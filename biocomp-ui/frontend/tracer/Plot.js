import React, { useEffect, useRef, useState, useMemo } from "react";
import ReactDOM from "react-dom";
import { layoutData, pointData } from "./data"; // import your data
import { COLORS } from "./constants";
import * as d3 from "d3";
import "./style.css";
import { select, line, scaleLinear, axisBottom, axisLeft } from "d3";
import Draggable, { DraggableCore } from "react-draggable";

/*──────────────────────────────▼     ScatterPlot    ▼──────────────────────────────*/

const ScatterPlot = ({ points, axisInfo, selected, width, height }) => {
  const ref = useRef();

  const getLabel = (axis) => {
    return `${axis.type} - ${axis.node_id}${axis.info ? " - " + axis.info : ""}`;
  };

  useEffect(() => {
    if (points && points.length > 0 && axisInfo) {
      const svg = d3.select(ref.current);

      // Clear existing SVG
      svg.selectAll("*").remove();

      let [xData, yData, cData] = [null, null, null];
      if (points.length <= 1) {
        return;
      } else if (points.length === 2) {
        [xData, yData] = points;
      } else if (points.length === 3) {
        [xData, yData, cData] = points;
      }

      const margin = { top: 60, right: 60, bottom: 60, left: 60 };
      const pwidth = width - margin.left - margin.right;
      const pheight = width - margin.top - margin.bottom;

      const x = d3.scaleLinear().domain(d3.extent(xData)).range([0, pwidth]);

      const y = d3.scaleLinear().domain(d3.extent(yData)).range([pheight, 0]);

      let color;
      if (cData) {
        color = d3.scaleSequential(d3.interpolateYlGnBu).domain(d3.extent(cData));
      }

      const plot = svg.append("g").attr("transform", `translate(${margin.left}, ${margin.top})`);

      plot
        .append("g")
        .attr("transform", `translate(0, ${pheight})`)
        .call(d3.axisBottom(x))
        .append("text")
        .attr("y", 40)
        .attr("x", pwidth / 2)
        .attr("fill", "black")
        .attr("text-anchor", "middle")
        .text(getLabel(axisInfo[0]));

      plot
        .append("g")
        .call(d3.axisLeft(y))
        .append("text")
        .attr("y", -40)
        .attr("x", -pheight / 2)
        .attr("fill", "black")
        .attr("transform", "rotate(-90)")
        .attr("text-anchor", "middle")
        .text(getLabel(axisInfo[1]));

      xData.forEach((xVal, i) => {
        const yVal = yData[i];
        const selectedPoint = selected[i];
        const point = plot.append("g").attr("transform", `translate(${x(xVal)}, ${y(yVal)})`);

        if (selectedPoint) {
          point
            .append("circle")
            .attr("r", 4)
            .attr("fill", cData ? color(cData[i]) : "black");
        } else {
          point
            .append("text")
            .text("+")
            .attr("fill", cData ? color(cData[i]) : "black");
        }
      });
    }
  }, [points, axisInfo, selected, width, height]);

  return <svg ref={ref} width={width} height={height} />;
};

/*════════════════════════════════════════════════════════════════════════════════*/

/*────────────────────────────▼     AxisSelector     ▼────────────────────────────*/

const AxisSelector = ({ axislist, label, setAxis }) => {
  const [selectedAxis, setSelectedAxis] = useState("None");

  useEffect(() => {
    if (selectedAxis !== "None") {
      setAxis(selectedAxis);
    }
  }, [selectedAxis, setAxis]);

  return (
    <div className="plot-axis-select-box">
      <select value={selectedAxis} onChange={(e) => setSelectedAxis(e.target.value)}>
        <option value="None">-- {label} --</option>
        {axislist.map((axis) => (
          <option key={axis.identifier} value={axis.identifier}>
            {axis.name}
          </option>
        ))}
      </select>
    </div>
  );
};

/*════════════════════════════════════════════════════════════════════════════════*/

//const Plot = ({ layoutData, pointData, pointInfo, rowColToIndexMap }) => {
const Plot = React.forwardRef(({ layoutData, pointData, pointInfo, rowColToIndexMap, position }, ref) => {
  /*────────────────────────────▼     getAxisInfo     ▼─────────────────────────────*/

  const getAxisInfo = (axisId) => {
    const rowCol = axisId.split("-");
    if (rowCol.length === 2) {
      const row = parseInt(rowCol[0]);
      const column = parseInt(rowCol[1]);
      return layoutData[row][column];
    }
    return null;
  };

  /*════════════════════════════════════════════════════════════════════════════════*/

  /*─────────────────────────────▼     getPoints     ▼──────────────────────────────*/

  const getPoints = (axisIdList) => {
    let values = [];
    console.log("axisIdList", axisIdList);
    axisIdList.forEach((axisId) => {
      const rowCol = axisId.split("-");
      if (rowCol.length === 2) {
        const row = parseInt(rowCol[0]);
        const column = parseInt(rowCol[1]);
        const idx = rowColToIndexMap[row][column];
        const points = pointInfo.points[idx];
        values.push(points.map((p) => p.value[0]));
      }
    });
    return values;
  };

  /*════════════════════════════════════════════════════════════════════════════════*/

  /*──────────────────────────────▼     axisList     ▼──────────────────────────────*/

  const axislist = useMemo(() => {
    const axislist = [];
    layoutData.forEach((row, rowIndex) => {
      row.forEach((item, columnIndex) => {
        axislist.push({
          name: `${item.type}-${item.node_id}-${item.slot}`,
          identifier: `${rowIndex}-${columnIndex}`,
        });
      });
    });
    return axislist;
  }, []);

  /*════════════════════════════════════════════════════════════════════════════════*/

  const [xAxis, setXAxis] = useState("None");
  const [yAxis, setYAxis] = useState("None");
  const [colorAxis, setColorAxis] = useState("None");
  const [points, setPoints] = useState([]);

  useEffect(() => {
    console.log("useEffect--------");
    console.log("xAxis", xAxis);
    console.log("yAxis", yAxis);
    console.log("colorAxis", colorAxis);

    setPoints(getPoints([xAxis, yAxis, colorAxis]));
  }, [xAxis, yAxis, colorAxis]);

  return (
    <Draggable>
      <div className="nopan plot-container" ref={ref}>
        <div className="nopan plot-axis-select">
          <AxisSelector axislist={axislist} label="X Axis" setAxis={setXAxis} />
          <AxisSelector axislist={axislist} label="Y Axis" setAxis={setYAxis} />
          <AxisSelector axislist={axislist} label="Color Axis" setAxis={setColorAxis} />
        </div>
        <div className="nopan plot">
          <ScatterPlot
            points={points}
            axisInfo={[getAxisInfo(xAxis), getAxisInfo(yAxis), getAxisInfo(colorAxis)]}
            selected={pointInfo.selectedTraces}
            width={500}
            height={500}
          />
        </div>
      </div>
    </Draggable>
  );
});

export default Plot;
