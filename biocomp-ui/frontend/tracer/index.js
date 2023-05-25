import React, { useEffect, useRef, useState, useMemo } from "react";
import ReactDOM from "react-dom";
import SliderAxis from "./SliderAxis";
import Menu from "./Menu";
import { layoutData, pointData } from "./data";
import { COLORS } from "./constants";
import * as d3 from "d3";
import "./style.css";
import dagre from "dagre";
import { select, line, scaleLinear, axisBottom, axisLeft } from "d3";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Plot from "./Plot";

const AXIS_OFFSET = { x: 170, y: 20 };
const AXIS_WIDTH = 200;

const NODE_WIDTH = 400;
const NODE_HEIGHT = 50;
const INV_NODE_HEIGHT = 0;

/*────────────────────────▼     plotting components     ▼─────────────────────────*/

const LinePlot = ({ points }) => {
  const width = 200;
  const height = 200;
  const ref = useRef();

  useEffect(() => {
    if (points && points.length > 0) {
      const svg = d3.select(ref.current);
      const xScale = d3
        .scaleLinear()
        .domain(d3.extent(points, (d) => d[0]))
        .range([0, width]);
      const yScale = d3
        .scaleLinear()
        .domain(d3.extent(points, (d) => d[1]))
        .range([height, 0]);

      const line = d3
        .line()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]));

      svg.append("path").data([points]).attr("d", line).attr("fill", "none").attr("stroke", "blue");
    }
  }, [points]);

  return <div ref={ref}></div>;
};

const ScatterPlot = ({ points }) => {
  const ref = useRef();
  const width = 200;
  const height = 200;

  useEffect(() => {
    if (points && points.length > 0) {
      const svg = d3.select(ref.current);
      const xScale = d3
        .scaleLinear()
        .domain(d3.extent(points, (d) => d[0]))
        .range([0, width]);
      const yScale = d3
        .scaleLinear()
        .domain(d3.extent(points, (d) => d[1]))
        .range([height, 0]);
      const colorScale = d3
        .scaleSequential()
        .domain(d3.extent(points, (d) => d[2]))
        .interpolator(d3.interpolateCool);

      svg
        .selectAll("circle")
        .data(points)
        .enter()
        .append("circle")
        .attr("cx", (d) => xScale(d[0]))
        .attr("cy", (d) => yScale(d[1]))
        .attr("r", 5)
        .attr("fill", (d) => colorScale(d[2]));
    }
  }, [points]);

  return <div ref={ref}></div>;
};

/*════════════════════════════════════════════════════════════════════════════════*/

function App() {
  const [settings, setSettings] = useState({ colorMode: "solid" });
  const [sceneTransform, setTransform] = useState();

  /*─────────────────────────────▼     RowCol map     ▼─────────────────────────────*/

  // maintain a map of row/column to index
  // so that we can easily find the index of a slider.
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

  const sliderRefs = useMemo(() => [], []);

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

  const init_selected = layoutData.map((row, rowIndex) => {
    return row.map((item, columnIndex) => {
      return false;
    });
  });

  const svgRef = useRef();

  /*─────────────────▼     build selected and position infos     ▼──────────────────*/

  const buildPointInfo = () => {
    let pnts = [];
    let selectedTraces = [];
    const svgRefCurrent = svgRef.current;

    const scene_state = {
      x: sceneTransform ? sceneTransform.state.positionX : 0,
      y: sceneTransform ? sceneTransform.state.positionY : 0,
      scale: sceneTransform ? sceneTransform.state.scale : 1,
    };

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
        const svgRect = svgRefCurrent.getBoundingClientRect();

        const axis_offset = {
          x: AXIS_OFFSET.x * scene_state.scale,
          y: AXIS_OFFSET.y * scene_state.scale,
        };

        const axis_width = AXIS_WIDTH * scene_state.scale;
        const pos = points.map((d, i) => {
          return {
            x: (rect.x + axis_offset.x + d * axis_width - svgRect.x) / scene_state.scale,
            y: (rect.y + axis_offset.y - svgRect.y) / scene_state.scale,
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

    let virtualNodes = {};
    for (let r = 0; r < layoutData.length; r++) {
      for (let c = 0; c < layoutData[r].length; c++) {
        let node = layoutData[r][c];
        //const height = node.type.startsWith("in") ? INV_NODE_HEIGHT : NODE_HEIGHT;
        const height = NODE_HEIGHT;
        g.setNode(`${r}-${c}`, { width: NODE_WIDTH, height: height });

        // we'll create virtual nodes for the ones that have multiple inputs
        if (node.n_inputs > 1) {
          const nid = node.node_id;
          virtualNodes[nid] = node;
        }
      }
    }

    // create virtual nodes
    for (let nid in virtualNodes) {
      const node_name = `virt-${nid}`;
      g.setNode(node_name, { width: 0, height: 0 });
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

        let target_nid = item.target_nid;
        // if in the virtual nodes, we'll link to the virtual node
        if (target_nid in virtualNodes) {
          target_nid = `virt-${target_nid}`;
          g.setEdge(`${r}-${c}`, target_nid);
          g.setEdge(target_nid, `${nextRow}-${nextColumn}`);
        } else {
          g.setEdge(`${r}-${c}`, `${nextRow}-${nextColumn}`); // Add edges to the graph
        }
      }
    }

    dagre.layout(g);

    const calculatedPositions = layoutData.map(() => []);
    g.nodes().forEach((nodeId) => {
      if (nodeId.startsWith("virt-")) return;

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

      case "solid":
      default:
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

  // we want to be able to add new plots by clicking a + button

  const [plotRefs, setPlotRefs] = useState([]);

  const addPlot = () => {
    const newPlotRefs = [...plotRefs];
    newPlotRefs.push(React.createRef());
    setPlotRefs(newPlotRefs);
  };

  const removePlot = (index) => {
    const newPlotRefs = [...plotRefs];
    newPlotRefs.splice(index, 1);
    setPlotRefs(newPlotRefs);
  };

  //Plot = ({ layoutData, pointData, pointInfo, rowColToIndexMap }) => {

  return (
    <>
      <button onClick={addPlot}>Add Plot</button>
      {plotRefs.map((ref, index) => (
        <Plot
          key={`plot-${index}`}
          ref={ref}
          layoutData={layoutData}
          pointData={pointData}
          pointInfo={pointInfo}
          rowColToIndexMap={rowColToIndexMap}
          position={{ x: index * 100, y: index * 100 }}
        />
      ))}

      <TransformWrapper
        minScale={0.4}
        maxScale={2.5}
        limitToBounds={false}
        panning={{
          excluded: [
            "plot-container",
            "nopan",
            "range-slider",
            "slider-track",
            "slider-thumb",
            "slideraxis",
            "plot",
            "plotcont",
            "react-draggable",
            "plot-axis-select",
          ],
        }}
        wheel={{ step: 0.05 }}
        onTransformed={setTransform}
        initialPositionX={-500}
        initialPositionY={-500}
      >
        {({ zoomIn, zoomOut, resetTransform, ...rest }) => (
          <>
            <TransformComponent>
              <svg className="traces" ref={svgRef} />

              {layoutData.map((row, rowIndex) =>
                row.map((item, columnIndex) => {
                  if (!sliderRefs[rowIndex]) sliderRefs[rowIndex] = [];
                  if (!sliderRefs[rowIndex][columnIndex])
                    sliderRefs[rowIndex][columnIndex] = React.createRef();
                  const { x, y } = calculatedPositions[rowIndex][columnIndex];
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
                })
              )}
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
