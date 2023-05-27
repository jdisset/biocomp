import React, { useEffect, useRef, useState, useMemo } from "react";
import ReactDOM from "react-dom/client";
import SliderAxis from "./SliderAxis";
import { COLORS } from "./constants";
import * as d3 from "d3";
import "./style.css";
import dagre from "dagre";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Plot from "./Plot";
import * as msgpack from "@msgpack/msgpack";
import fs from "fs";

const layoutDataBuffer = fs.readFileSync(__dirname + "/layoutData.bin");
const layoutData = msgpack.decode(new Uint8Array(layoutDataBuffer));
const pointDataBuffer = fs.readFileSync(__dirname + "/pointData.bin");
const pointData = msgpack.decode(new Uint8Array(pointDataBuffer));

const AXIS_OFFSET = { x: 170, y: 20 };
const AXIS_WIDTH = 200;

const NODE_WIDTH = 400;
const NODE_HEIGHT = 50;

const NSAMPLE = 50;

function App() {
  const [sceneTransform, setTransform] = useState();
  const sliderRefs = useMemo(() => [], []);

  const init_ranges = new Array(layoutData.length).fill([0, 1]);
  const [ranges, setRanges] = useState(init_ranges);

  const svgRef = useRef();

  /*─────────────▼     build position information for each point     ▼──────────────*/
  const buildPointInfo = () => {
    let pnts = [];
    const svgRefCurrent = svgRef.current;

    const scene_state = {
      x: sceneTransform ? sceneTransform.state.positionX : 0,
      y: sceneTransform ? sceneTransform.state.positionY : 0,
      scale: sceneTransform ? sceneTransform.state.scale : 1,
    };
    const axis_offset = {
      x: AXIS_OFFSET.x * scene_state.scale,
      y: AXIS_OFFSET.y * scene_state.scale,
    };
    const axis_width = AXIS_WIDTH * scene_state.scale;

    for (let idx = 0; idx < sliderRefs.length; idx++) {
      const sliderRef = sliderRefs[idx].current;
      if (!sliderRef) continue;
      const sliderRect = sliderRef.getBoundingClientRect();
      const svgRect = svgRefCurrent.getBoundingClientRect();

      const pos = pointData[idx].map((d, i) => {
        return {
          x: (sliderRect.x + axis_offset.x + d * axis_width - svgRect.x) / scene_state.scale,
          y: (sliderRect.y + axis_offset.y - svgRect.y) / scene_state.scale,
          node_uid: idx,
          value: d,
          i: i,
        };
      });
      pnts[idx] = pos;
    }
    return pnts;
  };

  const [pointInfo, setPointInfo] = useState(buildPointInfo());

  const buildSelectedTraces = () => {
    if (pointData.length === 0) return;
    if (ranges.length === 0) return;

    const NPOINTS = pointData[0].length;
    let selected = new Array(NPOINTS).fill(true);
    for (let idx = 0; idx < pointData.length; idx++) {
      const p = pointData[idx];
      const inRange = p.map((d, i) => {
        const range = ranges[idx];
        const inrange = (d >= range[0]) && (d <= range[1]);
        return inrange;
      });
      selected = selected.map((d, i) => d && inRange[i]);
    }
    return selected;
  };
  const [selectedTraces, setSelectedTraces] = useState();

  const buildVisibleTraces = () => {
    if (selectedTraces === undefined) return;
    // simply put the first NSAMPLES selected traces to true
    let visibleTraces = new Array(selectedTraces.length).fill(false);
    let n = 0;
    for (let idx = 0; idx < selectedTraces.length; idx++) {
      if (selectedTraces[idx]) {
        visibleTraces[idx] = true;
        n++;
      }
      if (n === NSAMPLE) break;
    }
    return visibleTraces;
  };
  const [visibleTraces, setVisibleTraces] = useState();

  const buildLineInfo = () => {
    if (!visibleTraces) return [];
    if (pointInfo.length === 0) return [];
    let allLines = [];
    for (let idx = 0; idx < layoutData.length; idx++) {
      const item = layoutData[idx];
      if (item.output_to.length === 0) continue;
      const next_idx = item.output_to[0];
      if (next_idx === undefined) continue;
      const pos1 = pointInfo[idx];
      const pos2 = pointInfo[next_idx];
      const linesData = pos1.map((p, i) => {
        if (visibleTraces[i])
          return {
            x1: p.x,
            y1: p.y,
            x2: pos2[i].x,
            y2: pos2[i].y,
            idx: idx,
            next_idx: next_idx,
            i: i,
          };
      });
      allLines = allLines.concat(linesData.filter((d) => d !== undefined));
    }
    return allLines;
  };

  /*════════════════════════════════════════════════════════════════════════════════*/

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

  const [isInitialRenderComplete, setIsInitialRenderComplete] = useState(false);

  const [lineInfo, setLineInfo] = useState(buildLineInfo());

  /*════════════════════════════════════════════════════════════════════════════════*/

  /*──────────────────────────────▼     drawing     ▼───────────────────────────────*/

  const [colorMode, setColorMode] = useState("solid"); // Default is 'solid'

  const getLineColor = (l, i) => {
    switch (colorMode) {
      case "solid":
      default:
        return selectedTraces[l.i] ? COLORS.selected_trace : COLORS.unselected_trace;
    }
  };

  const lineClass = (idx, nextidx, i) => `traceline-${idx}-${nextidx}-${i}`;

  const drawLines = (svg) => {
    if (!isInitialRenderComplete) return;
    if (pointInfo.length === 0) return;
    if (lineInfo.length === 0) return;
    if (calculatedPositions.length === 0) return;
    if (!visibleTraces) return;

    let filteredLineInfo = lineInfo.filter((l) => visibleTraces[l.i]);
    if (filteredLineInfo.length === 0) {
      svg.selectAll("line").remove();
      return;
    }
    let selectedLines = svg.selectAll("line").data(filteredLineInfo);
    selectedLines = selectedLines.enter().append("line").merge(selectedLines); // Apply to both new and existing lines
    selectedLines.exit().remove();
    selectedLines // Update all lines
      .attr("class", (l) => lineClass(l.idx, l.next_idx, l.i))
      .attr("x1", (d) => d.x1)
      .attr("y1", (d) => d.y1)
      .attr("x2", (d) => d.x2)
      .attr("y2", (d) => d.y2)
      .attr("stroke", getLineColor)
      .attr("stroke-width", (l) => 2)
      .raise();
    selectedLines.attr("stroke", (d, i) => getLineColor(d, i));
  };

  /*════════════════════════════════════════════════════════════════════════════════*/

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    drawLines(svg);
  }, [lineInfo, colorMode, isInitialRenderComplete, pointInfo, visibleTraces, calculatedPositions, selectedTraces, ranges]);

  useEffect(() => {
    setVisibleTraces(buildVisibleTraces());
  }, [selectedTraces, NSAMPLE, isInitialRenderComplete]);

  useEffect(() => {
    setSelectedTraces(buildSelectedTraces());
  }, [ranges, pointInfo, isInitialRenderComplete, calculatedPositions]);

  useEffect(() => {
    setCalculatedPositions(buildGraphLayout());
  }, [layoutData]); // update calculatedPositions whenever layoutData changes

  useEffect(() => {
    setPointInfo(buildPointInfo());
  }, [calculatedPositions, pointData, isInitialRenderComplete]);

  useEffect(() => {
    setLineInfo(buildLineInfo());
  }, [visibleTraces, isInitialRenderComplete]);

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

  const setSliderRange = (idx, range) => {
    // update selected traces when a slider is moved
    // called by Slider component
    //if (pointInfo.length === 0) return;
    //for (let i = 0; i < pointInfo[idx].length; i++) {
      //const p = pointInfo[idx][i];
      //const wasInRange = p.value >= ranges[idx][0] && p.value <= ranges[idx][1];
      //if (wasInRange && (p.value < range[0] || p.value > range[1])) becameOutOfFocus.push(i);
      //else if (wasInRange && p.value >= range[0] && p.value <= range[1]) becameInFocus.push(i);
    //}

    //let newSelectedTraces = [...selectedTraces];
    //for (let i = 0; i < becameInFocus.length; i++) newSelectedTraces[becameInFocus[i]] = true;
    //for (let i = 0; i < becameOutOfFocus.length; i++)
      //newSelectedTraces[becameOutOfFocus[i]] = false;

    setRanges((prevRanges) => {
      const newRanges = [...prevRanges];
      newRanges[idx] = range;
      return newRanges;
    });
    //setSelectedTraces(buildSelectedTraces());
    //setVisibleTraces(buildVisibleTraces());
  };

  // we want to be able to add new plots by clicking a + button
  //
  //
  //

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
                  selectedTraces={selectedTraces}
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

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
