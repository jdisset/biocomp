import React, { useEffect, useRef, useState, useMemo } from "react";
import ReactDOM from "react-dom";
import SliderAxis from "./SliderAxis";
import { layoutData, pointData } from "./data";
import { COLORS } from "./constants";
import * as d3 from "d3";
import "./style.css";
import dagre from "dagre";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Plot from "./Plot";

const AXIS_OFFSET = { x: 170, y: 20 };
const AXIS_WIDTH = 200;

const NODE_WIDTH = 400;
const NODE_HEIGHT = 50;

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
  const [sceneTransform, setTransform] = useState();
  const sliderRefs = useMemo(() => [], []);

  const init_ranges = new Array(layoutData.length).fill([0, 1]);
  const [ranges, setRanges] = useState(init_ranges);
  const setSliderRange = (idx, range) => {
    setRanges((prev) => {
      const newRanges = [...prev];
      newRanges[idx] = range;
      return newRanges;
    });
  };

  const svgRef = useRef();

  /*─────────────▼     build position information for each point     ▼──────────────*/

  const buildPointInfo = () => {
    let pnts = [];
    let selectedTraces = [];
    const svgRefCurrent = svgRef.current;

    const scene_state = {
      x: sceneTransform ? sceneTransform.state.positionX : 0,
      y: sceneTransform ? sceneTransform.state.positionY : 0,
      scale: sceneTransform ? sceneTransform.state.scale : 1,
    };

    for (let idx = 0; idx < sliderRefs.length; idx++) {
      const sliderRef = sliderRefs[idx].current;
      if (!sliderRef) continue;

      const minVal = ranges[idx][0];
      const maxVal = ranges[idx][1];

      const sliderRect = sliderRef.getBoundingClientRect();
      const svgRect = svgRefCurrent.getBoundingClientRect();

      const axis_offset = {
        x: AXIS_OFFSET.x * scene_state.scale,
        y: AXIS_OFFSET.y * scene_state.scale,
      };
      const axis_width = AXIS_WIDTH * scene_state.scale;

      const pos = pointData[idx].map((d, i) => {
        return {
          x: (sliderRect.x + axis_offset.x + d * axis_width - svgRect.x) / scene_state.scale,
          y: (sliderRect.y + axis_offset.y - svgRect.y) / scene_state.scale,
          node_uid: idx,
          inRange: d >= minVal && d <= maxVal,
          value: d,
          i: i,
        };
      });

      if (pnts.length <= idx) pnts.push(pos);
      else pnts[idx] = pos;

      const pointIsSelected = pos.map((d) => d.inRange);
      if (selectedTraces.length === 0) selectedTraces = pointIsSelected;
      selectedTraces = selectedTraces.map((d, i) => d && pointIsSelected[i]);
    }
    return { points: pnts, selectedTraces: selectedTraces };
  };

  /*════════════════════════════════════════════════════════════════════════════════*/

  const buildLineInfo = () => {
    if (pointInfo.points.length === 0) return [];
    let allLines = [];
    for (let idx = 0; idx < layoutData.length; idx++) {
      const item = layoutData[idx];
      if (item.output_to.length === 0) continue;
      const next_idx = item.output_to[0];
      if (next_idx === undefined) continue;
      const pos1 = pointInfo.points[idx];
      const pos2 = pointInfo.points[next_idx];
      const linesData = pos1.map((p, i) => ({
        x1: p.x,
        y1: p.y,
        x2: pos2[i].x,
        y2: pos2[i].y,
        idx: idx,
        next_idx: next_idx,
        i: i,
      }));
      allLines = allLines.concat(linesData);
    }
    return allLines;
  };

  const buildGraphLayout = () => {
    var g = new dagre.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel(() => ({}));

    let virtualNodes = {};
    // for each node, idx in layout data:
    layoutData.forEach((node, idx) => {
      const height = NODE_HEIGHT;
      g.setNode(idx, { width: NODE_WIDTH, height: height });
      // we'll create virtual nodes for the ones that have multiple inputs
      if (node.n_inputs > 1) {
        const nid = node.node_id;
        virtualNodes[nid] = node;
      }
    });

    // create virtual nodes
    for (let nid in virtualNodes) {
      const node_name = `virt-${nid}`;
      g.setNode(node_name, { width: 0, height: 0 });
    }

    // Next, add edges to the graph.
    layoutData.forEach((node, idx) => {
      if (node.output_to.length === 0) return;

      const next_idx = node.output_to[0];
      if (next_idx === undefined) return;

      let target_nid = node.target_nid;
      if (target_nid in virtualNodes) {
        target_nid = `virt-${target_nid}`;
        g.setEdge(idx, target_nid);
        g.setEdge(target_nid, next_idx);
      } else {
        g.setEdge(idx, next_idx);
      }
    });

    dagre.layout(g);

    const calculatedPositions = layoutData.map(() => []);
    g.nodes().forEach((nodeId) => {
      if (nodeId.startsWith("virt-")) return;
      const node = g.node(nodeId);
      const idx = parseInt(nodeId);
      calculatedPositions[idx] = { x: node.x, y: node.y };
    });
    return calculatedPositions;
  };

  const [calculatedPositions, setCalculatedPositions] = useState(buildGraphLayout());

  useEffect(() => {
    setCalculatedPositions(buildGraphLayout());
  }, [layoutData]); // update calculatedPositions whenever layoutData changes

  const [isInitialRenderComplete, setIsInitialRenderComplete] = useState(false);

  const pointInfo = useMemo(
    () => buildPointInfo(),
    [pointData, ranges, sliderRefs, isInitialRenderComplete]
  );

  const lineInfo = useMemo(() => buildLineInfo(), [pointInfo, isInitialRenderComplete]);

  useEffect(() => {
    for (let idx = 0; idx < sliderRefs.length; idx++) {
      const ref = sliderRefs[idx];
      if (!ref.current) {
        setIsInitialRenderComplete(false);
        return;
      }
    }
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

    // in order to filter only the selected traces, we have acess to pointInfo.selectedTraces
    // which is an array of booleans, one for each trace.
    // we can use this to filter the gradients
    //

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
        const idx1 = d.idx;
        const v1 = pointInfo.points[idx1][d.i].value[0];
        const idx2 = d.next_idx;
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

  const lineClass = (idx, nextidx, i) => `traceline-${idx}-${nextidx}-${i}`;

  const drawLines = (svg) => {
    let lines = svg.selectAll("line").data(lineInfo);

    lines = lines.enter().append("line").merge(lines); // Apply to both new and existing lines

    lines // Update all lines
      .attr("class", (l, i) => lineClass(l.idx, l.next_idx, i))
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

  return (
    <>
      <button onClick={addPlot}>Add Plot</button>

      <TransformWrapper
        minScale={0.4}
        maxScale={2.5}
        limitToBounds={false}
        panning={{
          excluded: [
            "plot-container",
            "range-slider",
            "slider-track",
            "slider-thumb",
            "slideraxis",
            "plot",
            "plotcont",
            "react-draggable",
            "select",
            "plot-axis-select",
            "plot-ctrl-row",
            "plot-ctrls",
            "plot-select",
            "plot-select-box",
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

              {layoutData.map((item, idx) => {
                if (!sliderRefs[idx]) sliderRefs[idx] = React.createRef();
                const { x, y } = calculatedPositions[idx];
                return (
                  <SliderAxis
                    key={`slider-${idx}`}
                    sliderData={item}
                    scale={sceneTransform && sceneTransform.state ? sceneTransform.state.scale : 1}
                    points={pointData[idx]}
                    ref={sliderRefs[idx]}
                    setSliderRange={(range) => setSliderRange(idx, range)}
                    style={{ position: "absolute", left: x, top: y }}
                  />
                );
              })}

              {plotRefs.map((ref, index) => (
                <Plot
                  key={`plot-${index}`}
                  ref={ref}
                  layoutData={layoutData}
                  pointInfo={pointInfo}
                  scale={sceneTransform && sceneTransform.state ? sceneTransform.state.scale : 1}
                  removePlot={() => removePlot(index)}
                />
              ))}
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
