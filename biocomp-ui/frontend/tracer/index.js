import React, { useEffect, useRef, useState, useMemo } from "react";
import ReactDOM from "react-dom";
import SliderAxis from "./SliderAxis"; // import the SliderAxis component
import { layoutData, pointData } from "./data"; // import your data
import { COLORS } from "./constants";
import * as d3 from "d3";
import "./style.css";
import dagre from "dagre";

const AXIS_OFFSET = { x: 150, y: 20 };
const AXIS_WIDTH = 200;

function App() {
  //const sliderRefs = layoutData.map((row, rowIndex) => {
  //return row.map((item, columnIndex) => {
  //return React.createRef();
  //});
  //});

  const sliderRefs = useMemo(() => [], []);

  // create ranges in a 2D list
  const init_ranges = layoutData.map((row, rowIndex) => {
    return row.map((item, columnIndex) => {
      return [0, 1];
    });
  });
  console.log(layoutData);

  const [ranges, setRanges] = useState(init_ranges);

  const setSliderRange = (rowIndex, columnIndex, range) => {
    setRanges((prev) => {
      const newRanges = [...prev];
      newRanges[rowIndex][columnIndex] = range;
      return newRanges;
    });
  };

  const svgRef = useRef();

  /*─────────────────────────────▼     RowCol map     ▼─────────────────────────────*/

  // maintain a map of row/column to index
  // so that we can easily find the index of a slider.
  // We want to update it if the layoutData changes
  const buildRowColToIndexMap = () => {
    let index = 0;
    let rowColToIndexMap = new Array(layoutData.length);
    for (let rowIndex in layoutData) {
      for (let columnIndex in layoutData[rowIndex]) {
        rowColToIndexMap[rowIndex] = rowColToIndexMap[rowIndex] || [];
        rowColToIndexMap[rowIndex][columnIndex] = index;
        index++;
      }
    }
    return rowColToIndexMap;
  };

  const [rowColToIndexMap, setRowColToIndexMap] = useState(buildRowColToIndexMap());

  useEffect(() => {
    setRowColToIndexMap(buildRowColToIndexMap());
  }, [layoutData]);

  /*════════════════════════════════════════════════════════════════════════════════*/

  /*─────────────────▼     build selected and position infos     ▼──────────────────*/

  const buildPointInfo = () => {
    let pnts = [];
    let selectedTraces = [];
    for (const rowIndex in pointData) {
      for (const columnIndex in pointData[rowIndex]) {
        if (
          sliderRefs[rowIndex] === undefined ||
          sliderRefs[rowIndex][columnIndex] === undefined ||
          !sliderRefs[rowIndex][columnIndex].current
        ) {
          continue;
        }
        const ref = sliderRefs[rowIndex][columnIndex];

        const rect = ref.current.getBoundingClientRect();

        const minVal = ranges[rowIndex][columnIndex][0];
        const maxVal = ranges[rowIndex][columnIndex][1];
        const points = pointData[rowIndex][columnIndex];

        const pos = points.map((d, i) => {
          return {
            x: rect.x + AXIS_OFFSET.x + d * AXIS_WIDTH + window.scrollX,
            y: rect.y + AXIS_OFFSET.y + window.scrollY,
            row: rowIndex,
            column: columnIndex,
            inRange: d >= minVal && d <= maxVal,
            value: d,
            i: i,
          };
        });

        const idx = rowColToIndexMap[rowIndex][columnIndex];
        if (pnts.length <= idx) {
          pnts.push(pos);
        } else {
          pnts[idx] = pos;
        }

        const pointIsSelected = pos.map((d) => d.inRange);
        if (selectedTraces.length === 0) {
          selectedTraces = pointIsSelected;
        }
        selectedTraces = selectedTraces.map((d, i) => d && pointIsSelected[i]);
      }
    }
    return { points: pnts, selectedTraces: selectedTraces };
  };

  const buildLineInfo = () => {
    if (pointInfo.points.length === 0) {
      return [];
    }
    let allLines = [];
    for (let r = 0; r < layoutData.length; r++) {
      for (let c = 0; c < layoutData[r].length; c++) {
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

        const this_idx = rowColToIndexMap[r][c];
        const next_idx = rowColToIndexMap[nextRow][nextColumn];
        //console.log(pointInfo);
        const pos1 = pointInfo.points[this_idx];
        const pos2 = pointInfo.points[next_idx];
        const linesData = pos1.map((p, i) => ({
          x1: p.x,
          y1: p.y,
          x2: pos2[i].x,
          y2: pos2[i].y,
          row: r,
          column: c,
          nextRow: nextRow,
          nextColumn: nextColumn,
          i: i,
        }));
        allLines = allLines.concat(linesData);
      }
    }
    return allLines;
  };

  const buildCalculatedPositions = () => {
    var g = new dagre.graphlib.Graph();

    g.setGraph({});

    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(() => ({}));

    for (let r = 0; r < layoutData.length; r++) {
      for (let c = 0; c < layoutData[r].length; c++) {
        g.setNode(`${r}-${c}`, { width: 300, height: 50 });
      }
    }

    // Next, add edges to the graph.
    for (let r = 0; r < layoutData.length; r++) {
      for (let c = 0; c < layoutData[r].length; c++) {
        const item = layoutData[r][c];
        if (item.output_to.length === 0) {
          continue;
        }

        const nextRow = item.output_to[0][0];
        const nextColumn = item.output_to[0][1];
        if (nextRow === undefined || nextColumn === undefined) {
          continue;
        }

        g.setEdge(`${r}-${c}`, `${nextRow}-${nextColumn}`); // Add edges to the graph
      }
    }

    dagre.layout(g);

    const calculatedPositions = layoutData.map(() => []);
    g.nodes().forEach((nodeId) => {
      const node = g.node(nodeId);
      const [row, col] = nodeId.split("-").map(Number);

      calculatedPositions[row][col] = { x: node.x, y: node.y };
    });

    return calculatedPositions;
  };

  const [calculatedPositions, setCalculatedPositions] = useState(buildCalculatedPositions());

  useEffect(() => {
    setCalculatedPositions(buildCalculatedPositions());
  }, [layoutData]); // update calculatedPositions whenever layoutData changes

  const [isInitialRenderComplete, setIsInitialRenderComplete] = useState(false);

  const pointInfo = useMemo(
    () => buildPointInfo(),
    [pointData, ranges, sliderRefs, isInitialRenderComplete]
  );

  const lineInfo = useMemo(() => buildLineInfo(), [pointInfo, isInitialRenderComplete]);

  useEffect(() => {
    for (let row of sliderRefs) {
      for (let ref of row) {
        if (!ref.current) {
          return; // Exit early if a ref is not set
        }
      }
    }
    // If this point is reached, all refs have been set
    setIsInitialRenderComplete(true);
  }, [sliderRefs]);

  /*════════════════════════════════════════════════════════════════════════════════*/

  /*──────────────────────────────▼     drawing     ▼───────────────────────────────*/

  const [colorMode, setColorMode] = useState("gradient"); // Default is 'solid'

  const getLineColor = (l, i) => {
    switch (colorMode) {
      case "gradient":
        if (pointInfo.selectedTraces[l.i]) {
          return `url(#gradient-${i})`; // Referencing the gradient
        } else {
          return COLORS.unselected_trace;
        }

      case "selectedAxis":
      // Use a color depending on the selected axis
      // Again, this is a simplified example
      //const selectedAxisColor = [>...<];
      //return selectedAxisColor;

      case "solid":
      default:
        // By default, use solid color
        return pointInfo.selectedTraces[l.i] ? COLORS.selected_trace : COLORS.unselected_trace;
    }
  };

  const generateGradients = (svg) => {
    const gradients = svg.selectAll("linearGradient").data(lineInfo, (d, i) => i);

    gradients
      .enter()
      .append("linearGradient")
      .attr("id", (d, i) => `gradient-${i}`) // This id is referenced in getLineColor
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", (d) => d.x1)
      .attr("y1", (d) => d.y1)
      .attr("x2", (d) => d.x2)
      .attr("y2", (d) => d.y2)
      .selectAll("stop")
      .data((d) => {
        const idx1 = rowColToIndexMap[d.row][d.column];
        const v1 = pointInfo.points[idx1][d.i].value[0];
        const idx2 = rowColToIndexMap[d.nextRow][d.nextColumn];
        const v2 = pointInfo.points[idx2][d.i].value[0];
        const cmap = d3.scaleSequential(d3.interpolateYlGnBu).domain([0, 1]);
        return [
          { offset: "0%", color: cmap(v1) },
          { offset: "50%", color: cmap((v1 + v2) / 2) },
          { offset: "100%", color: cmap(v2) },
        ];
      })
      .enter()
      .append("stop")
      .attr("offset", (d) => d.offset)
      .attr("stop-color", (d) => d.color);

    gradients.exit().remove(); // Remove unused gradients
  };

  useEffect(() => {
    if (!isInitialRenderComplete) return;
    const svg = d3.select(svgRef.current);
    if (colorMode === "gradient") generateGradients(svg);
  }, [colorMode, isInitialRenderComplete, pointInfo, lineInfo]);

  const oldKeys = useRef([]);

  const lineClass = (r, c, nextRow, nextColumn, i) =>
    `traceline-${r}-${c}-${nextRow}-${nextColumn}-${i}`;

  const drawLines = (svg) => {

    let lines = svg.selectAll("line").data(lineInfo);

    lines = lines.enter().append("line").merge(lines); // Apply to both new and existing lines

    lines // Update all lines
      .attr("class", (l, i) => lineClass(l.row, l.column, l.nextRow, l.nextColumn, l.i))
      .attr("x1", (d) => d.x1)
      .attr("y1", (d) => d.y1)
      .attr("x2", (d) => d.x2)
      .attr("y2", (d) => d.y2)
      .attr("stroke", getLineColor)
      .attr("stroke-width", (l) => (pointInfo.selectedTraces[l.i] ? 2 : 0.5));

    // to raise only the selected traces (and leave the unselected traces behind):
    lines.filter((l) => pointInfo.selectedTraces[l.i]).raise();

    lines.exit().remove(); // Remove old elements

    // UPDATE existing elements.
    lines.attr("stroke", (d, i) => getLineColor(d, i));
  };

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    drawLines(svg);
  }, [pointInfo, lineInfo, colorMode]);

  /*════════════════════════════════════════════════════════════════════════════════*/

  return (
    <div>
      <svg className="traces" ref={svgRef} />
      {layoutData.map((row, rowIndex) => (
        <div key={`row-${rowIndex}`} style={{ display: "flex", flexDirection: "row" }}>
          {row.map((item, columnIndex) => {
            if (!sliderRefs[rowIndex]) {
              sliderRefs[rowIndex] = [];
            }
            if (!sliderRefs[rowIndex][columnIndex]) {
              sliderRefs[rowIndex][columnIndex] = React.createRef();
            }
            const { x, y } = calculatedPositions[rowIndex][columnIndex]; // assuming calculatedPositions is the result of your dagre layout
            return (
              <SliderAxis
                key={`slider-${rowIndex}-${columnIndex}`}
                sliderData={item}
                points={pointData[rowIndex][columnIndex]}
                ref={sliderRefs[rowIndex][columnIndex]}
                setSliderRange={(range) => setSliderRange(rowIndex, columnIndex, range)}
                style={{ position: "absolute", left: x, top: y }}
              />
            );
          })}
        </div>
      ))}
      <div className="settings">
        <select value={colorMode} onChange={(e) => setColorMode(e.target.value)}>
          <option value="solid">Solid</option>
          <option value="gradient">Gradient</option>
          <option value="selectedAxis">Selected Axis</option>
        </select>
      </div>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
