import React, { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import "./style.css";
import Draggable from "react-draggable";

const CMAPS = {
  YlGnBu: d3.interpolateYlGnBu,
  Viridis: d3.interpolateViridis,
  Inferno: d3.interpolateInferno,
  YlOrRd: d3.interpolateYlOrRd,
  Blues: d3.interpolateBlues,
};

/*──────────────────────────────▼     ScatterPlot    ▼──────────────────────────────*/

const ScatterPlot = ({ points, axisInfo, selected, width, height, colorScale }) => {
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
        const cmap = colorScale ? CMAPS[colorScale] : d3.interpolateYlGnBu;
        color = d3.scaleSequential(cmap).domain(d3.extent(cData));
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
            .attr("fill", cData ? color(cData[i]) : "black").raise();
        } else {
          point
            .append("circle")
            .attr("r", 1)
            .attr("fill", "#aaaa").lower();
        }
      });

      if (cData) {
        // Colorbar position and dimensions
        const colorbarX = pwidth + 10;
        const colorbarY = 0;
        const colorbarWidth = 20;
        const colorbarHeight = pheight;

        // Create a gradient for the colorbar
        const colorbarGradient = svg
          .append("defs")
          .append("linearGradient")
          .attr("id", "colorbarGradient")
          .attr("x1", "0%")
          .attr("y1", "100%")
          .attr("x2", "0%")
          .attr("y2", "0%");

        // Add color stops to the gradient
        const gradientSteps = 10; // Number of steps in the gradient
        const colorDomain = d3.extent(cData);
        for (let i = 0; i <= gradientSteps; i++) {
          const value = colorDomain[0] + (colorDomain[1] - colorDomain[0]) * (i / gradientSteps);
          colorbarGradient
            .append("stop")
            .attr("offset", `${100 * (i / gradientSteps)}%`)
            .attr("stop-color", color(value));
        }

        // Add the colorbar
        plot
          .append("rect")
          .attr("x", colorbarX)
          .attr("y", colorbarY)
          .attr("width", colorbarWidth)
          .attr("height", colorbarHeight)
          .style("fill", "url(#colorbarGradient)");

        // Add an axis for the colorbar
        const colorScale = d3.scaleLinear().domain(colorDomain).range([colorbarHeight, 0]);

        const colorAxis = d3.axisRight(colorScale);

        plot
          .append("g")
          .attr("transform", `translate(${colorbarX + colorbarWidth}, ${colorbarY})`)
          .call(colorAxis);
      }
    }
  }, [points, axisInfo, selected, width, height, colorScale]);

  return <svg ref={ref} width={width} height={height} className="plot" />;
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

  const orderedAxisList = useMemo(() => {
    return axislist.sort((a,b) => a.node_id - b.node_id);
  }, [axislist]);

  return (
    <div className="plot-select-box">
      <select className="plot-select"
        value={selectedAxis} onChange={(e) => setSelectedAxis(e.target.value)}>
        <option value="None">-- {label} --</option>
        {orderedAxisList.map((axis) => (
          <option value={axis.identifier}>
            {axis.name}
          </option>
        ))}
      </select>
    </div>
  );
};

/*════════════════════════════════════════════════════════════════════════════════*/

const Plot = React.forwardRef(({ layoutData, pointInfo, selectedTraces, removePlot, scale }, ref) => {
  /*────────────────────────────▼     getInfo     ▼─────────────────────────────*/
  const axislist = useMemo(() => {
    return layoutData.map((item, idx) => ({
      name: `${item.node_id} ${item.type} ${item.n_outputs > 1 ? "[" + item.slot + "]" : ""}`,
      identifier: `${idx}`,
      node_id: item.node_id,
    }));
  }, []);

  const getAxisInfo = (axisId) => {
    if (axisId === "None") return null;
    const idx = parseInt(axisId);
    return layoutData[idx];
  };

  const getPoints = (axisIdList) => {
    let values = [];
    axisIdList.forEach((axisId) => {
      if (axisId !== "None") {
        const idx = parseInt(axisId);
        const points = pointInfo[idx];
        values.push(points.map((p) => p.value[0]));
      }
    });
    return values;
  };

  /*════════════════════════════════════════════════════════════════════════════════*/

  const [xAxis, setXAxis] = useState("None");
  const [yAxis, setYAxis] = useState("None");
  const [colorAxis, setColorAxis] = useState("None");
  const [points, setPoints] = useState([]);
  const [colorScale, setColorScale] = useState(null);

  useEffect(() => {
    setPoints(getPoints([xAxis, yAxis, colorAxis]));
  }, [xAxis, yAxis, colorAxis]);

  return (
    <Draggable scale={scale}>
      <div className="plot-container" ref={ref}>
        <div className="plot-remove-button" onClick={removePlot}></div>
        <div className="plot-ctrl-row">
          <AxisSelector axislist={axislist} label="X Axis" setAxis={setXAxis} />
          <AxisSelector axislist={axislist} label="Y Axis" setAxis={setYAxis} />
          <AxisSelector axislist={axislist} label="Color Axis" setAxis={setColorAxis} />
        </div>
        <div className="plot-ctrl-row">
          <div className="plot-select-box">
            <select onChange={(e) => setColorScale(e.target.value)}>
              {Object.keys(CMAPS).map((cmap) => (
                <option key={cmap} value={cmap}>
                  {cmap}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="plot">
          <ScatterPlot
            points={points}
            axisInfo={[getAxisInfo(xAxis), getAxisInfo(yAxis), getAxisInfo(colorAxis)]}
            selected={selectedTraces}
            colorScale={colorScale}
            width={480}
            height={480}
          />
        </div>
      </div>
    </Draggable>
  );
});

export default Plot;
