import React, { useState } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";

function AxisSlider({ values, activeRange, name }) {
  const [min, max] = activeRange;

  const dots = values.map((value, i) => (
    <circle
      key={i}
      cx={100 * (1 - value)}
      cy={50}
      r={2}
      fill={value >= min && value <= max ? "red" : "grey"}
    />
  ));

  return (
    <g transform={`translate(0, ${name * 200})`} className="axis-slider">
      <text x={-10} y={50} textAnchor="end">{`f${name}`}</text>
      {dots}
    </g>
  );
}

function ParallelCoordinates({ data, layout }) {
  const [activeRanges, setActiveRanges] = useState(layout.flat().map(() => [0, 1]));

  const lines = data.map((datapoint, i) => (
    <path
      key={i}
      d={d3.line()(layout.flat().map((f, j) => [100 * (1 - datapoint[f]), 50 * f]))}
      stroke={activeRanges.every(([min, max], j) => datapoint[layout.flat()[j]] >= min && datapoint[layout.flat()[j]] <= max) ? "red" : "grey"}
      fill="none"
    />
  ));

  return (
    <svg width="200" height={200 * layout.length}>
      {lines}
      {layout.flat().map((f, i) => (
        <AxisSlider
          key={i}
          values={data.map(datapoint => datapoint[f])}
          activeRange={activeRanges[i]}
          name={i}
        />
      ))}
    </svg>
  );
}

function App() {
  const f1 = () => Math.random();
  const f2 = v => (v > 0.5 ? v / 2 : v * 2) % 1;
  const f3 = v => 1 - v;
  const f4 = (v1, v2) => (v1 + v2) / 2;

  const data = Array.from({ length: 100 }, () => {
    const value1 = f1();
    const value2 = f2(value1);
    const value3 = f3(value2);
    const value4 = f4(value2, value3);

    return [value1, value2, value3, value4];
  });

  return <ParallelCoordinates data={data} layout={[[0], [1, 2], [3]]} />;
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
