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
import shortid from "shortid";
import fs from "fs";

import { useWhatChanged } from "@simbathesailor/use-what-changed";

const layoutDataBuffer = fs.readFileSync(__dirname + "/layoutData.bin");

const layoutInfo = msgpack.decode(new Uint8Array(layoutDataBuffer));
const layoutData = layoutInfo.layout;
const networkName = layoutInfo.network_name;
const pointDataBuffer = fs.readFileSync(__dirname + "/pointData.bin");
const pointData = msgpack.decode(new Uint8Array(pointDataBuffer));
const NTRACES = pointData.length === 0 ? 0 : pointData[0].length;
const NNODES = layoutData.length;
const AXIS_OFFSET = { x: 170, y: 20 };
const AXIS_WIDTH = 200;

const NODE_WIDTH = 400;
const NODE_HEIGHT = 50;

const NSAMPLE_IN = 70;
const NSAMPLE_OUT = 30;

const init_ranges = () => {
  let ranges = [];
  for (let i = 0; i < NNODES; i++) {
    const min = 0;
    const max = Math.max(...pointData[i]);

    ranges.push([min, Math.max(1, max)]);
  }
  return ranges;
};
const initial_ranges = init_ranges();

function App() {
  const [sceneTransform, setTransform] = useState();
  const sliderRefs = useMemo(() => [], []);

  const [ranges, setRanges] = useState(initial_ranges);

  const svgRef = useRef();

  /*─────────────▼     build position information for each point     ▼──────────────*/
  const buildPointInfo = () => {
    console.log("buildPointInfo");
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

    for (let idx = 0; idx < sliderRefs.length; idx++) {
      const range_len = initial_ranges[idx][1] - initial_ranges[idx][0];
      const axis_width = (AXIS_WIDTH * scene_state.scale) / range_len;
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

  const [pointInfo, setPointInfo] = useState();

  const [selectedCounts, setSelectedCounts] = useState();

  const buildSelectedCounts = () => {
    console.log("buildSelectedCounts");
    if (pointData.length === 0) return;
    if (ranges.length === 0) return;

    let selected = Array.from({ length: NTRACES }, () =>
      Array.from({ length: NNODES }, () => false)
    );
    for (let idx = 0; idx < pointData.length; idx++) {
      const p = pointData[idx];
      const range = ranges[idx];
      const inRange = p.map((d) => {
        const inrange = d >= range[0] && d <= range[1];
        return inrange;
      });
      for (let i = 0; i < inRange.length; i++) {
        selected[i][idx] = inRange[i];
      }
    }
    return selected;
  };

  const buildAllSelectedTraces = () => {
    console.log("buildAllSelectedTraces");
    if (selectedCounts === undefined) return;
    const selected = selectedCounts.map((d) => d.every((dd) => dd));
    return selected;
  };

  const [selectedTraces, setSelectedTraces] = useState();

  const buildVisibleTraces = () => {
    console.log("buildVisibleTraces");
    if (selectedTraces === undefined) return;
    // simply put the first NSAMPLES selected traces to true
    let visibleTraces = new Array(selectedTraces.length).fill(false);
    let n_in = 0;
    let n_out = 0;
    for (let idx = 0; idx < selectedTraces.length; idx++) {
      if (selectedTraces[idx] && n_in < NSAMPLE_IN) {
        visibleTraces[idx] = true;
        n_in++;
      }
      if (!selectedTraces[idx] && n_out < NSAMPLE_OUT) {
        visibleTraces[idx] = true;
        n_out++;
      }
      if (n_in >= NSAMPLE_IN && n_out >= NSAMPLE_OUT) break;
    }
    return visibleTraces;
  };

  const [visibleTraces, setVisibleTraces] = useState();

  const buildLineInfo = () => {
    console.log("buildLineInfo");
    if (!visibleTraces) return [];
    if (pointInfo.length === 0) return [];

    // filter out invisible traces
    let vtraces = [];
    visibleTraces.map((d, i) => {
      if (d) vtraces.push(i);
    });

    let allLines = [];
    for (let idx = 0; idx < layoutData.length; idx++) {
      const item = layoutData[idx];
      if (item.output_to.length === 0) continue;
      const next_idx = item.output_to[0];
      if (next_idx === undefined) continue;
      const linesData = vtraces.map((i) => {
        const p1 = pointInfo[idx][i];
        const p2 = pointInfo[next_idx][i];
        return {
          x1: p1.x,
          y1: p1.y,
          x2: p2.x,
          y2: p2.y,
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
    console.log("buildGraphLayout");
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

  // use a memo to avoid re-calculating the layoutk
  const calculatedPositions = useMemo(() => buildGraphLayout(), [layoutData]);

  const [isInitialRenderComplete, setIsInitialRenderComplete] = useState(false);

  const [lineInfo, setLineInfo] = useState();

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
    console.log("drawLines");
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
  }, [lineInfo, colorMode, isInitialRenderComplete]);

  useEffect(() => {
    setVisibleTraces(buildVisibleTraces());
  }, [selectedTraces, NSAMPLE_IN, NSAMPLE_OUT, isInitialRenderComplete]);

  useEffect(() => {
    setSelectedCounts(buildSelectedCounts());
    setSelectedTraces(buildAllSelectedTraces());
  }, [isInitialRenderComplete, calculatedPositions]);

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
    if (!pointInfo || pointInfo.length === 0) return;

    let becameInFocus = [];
    let becameOutOfFocus = [];

    for (let i = 0; i < pointInfo[idx].length; i++) {
      const p = pointInfo[idx][i];
      const wasInRange = p.value >= ranges[idx][0] && p.value <= ranges[idx][1];
      const isInRange = p.value >= range[0] && p.value <= range[1];
      if (wasInRange && !isInRange) {
        becameOutOfFocus.push(i);
      }
      if (!wasInRange && isInRange) {
        becameInFocus.push(i);
      }
    }
    let newSelectedTraces = [...selectedTraces];
    let newSelectedCounts = [...selectedCounts];
    for (let i = 0; i < becameInFocus.length; i++) {
      const p = pointInfo[idx][becameInFocus[i]];
      newSelectedCounts[p.i][idx] = true;
      newSelectedTraces[p.i] = newSelectedCounts[p.i].every((v) => v);
    }
    for (let i = 0; i < becameOutOfFocus.length; i++) {
      const p = pointInfo[idx][becameOutOfFocus[i]];
      newSelectedCounts[p.i][idx] = false;
      newSelectedTraces[p.i] = false;
    }

    setSelectedCounts(newSelectedCounts);
    setSelectedTraces(newSelectedTraces);

    let newRanges = [...ranges];
    newRanges[idx] = range;
    setRanges(newRanges);
  };

  const [bbox, setBbox] = useState({ x: 0, y: 0, width: 0, height: 0 });
  useEffect(() => {
    let max_x = 0;
    let max_y = 0;
    for (let i = 0; i < calculatedPositions.length; i++) {
      const p = calculatedPositions[i];
      max_x = Math.max(max_x, p.x);
      max_y = Math.max(max_y, p.y);
    }
    setBbox({ x: 0, y: 0, width: max_x + 500, height: max_y + 500 });
  }, [calculatedPositions]);

  const [plots, setPlots] = useState([]);

  const addPlot = () => {
    setPlots((prevPlots) => [...prevPlots, { id: shortid.generate(), ref: React.createRef() }]);
  };

  const removePlot = (id) => {
    setPlots((prevPlots) => prevPlots.filter((plot) => plot.id !== id));
  };

  return (
    <>
      <button onClick={addPlot} className="add-plot-btn">
        {" "}
        Add Plot
      </button>

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
              <div
                className="network-info"
                style={{ position: "absolute", top: -300, left: bbox.width / 3 }}
              >
                 "{networkName}" ({NTRACES} traces)
              </div>
              <svg className="traces" ref={svgRef} width={bbox.width} height={bbox.height} />

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
                    min={initial_ranges[idx][0]}
                    max={initial_ranges[idx][1]}
                    setSliderRange={(range) => setSliderRange(idx, range)}
                    style={{ position: "absolute", left: x, top: y }}
                  />
                );
              })}

              {plots.map((plot) => (
                <Plot
                  key={plot.id}
                  ref={plot.ref}
                  layoutData={layoutData}
                  pointInfo={pointInfo}
                  selectedTraces={selectedTraces}
                  defaultPosition={{ x: bbox.width / 3, y: bbox.height / 3 }}
                  scale={sceneTransform && sceneTransform.state ? sceneTransform.state.scale : 1}
                  removePlot={() => removePlot(plot.id)}
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
