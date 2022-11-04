import React from "react";
import axios from "axios";
import SRCNode from "./SRCNode";
import { useState, useEffect, useRef, useCallback } from "react";
import { generateArcs } from "./shapes.jsx";
import Util from "./util.jsx";
import "./style_app.css";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
} from "reactflow";

// PLASMID COMPONENT
function Plasmid(props) {
  const circle_radius = 15;
  const center = circle_radius + 2;
  let arcs = generateArcs(props.data.output_to.length, circle_radius, center);

  const text = (
    <text fill="black" fontSize="12" letterSpacing="0.05em">
      <tspan x={center} y={center + 2} textAnchor="middle">
        {props.data.type}
      </tspan>
    </text>
  );

  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div className="plasmid" key={'p'+props.data.source_id}>
      <svg
        width={(circle_radius + 2) * 2}
        height={(circle_radius + 2) * 2}
        viewBox={`0 0 ${circle_radius * 2 + 4} ${circle_radius * 2 + 4}`}
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx={center} cy={center} r={circle_radius + 1.0} fill="white" />
        <circle cx={center} cy={center} r={circle_radius} stroke="#EEEEEE" strokeWidth="3" />
        {arcs}
        <circle cx={center} cy={center} r={circle_radius + 1.5} stroke="black" strokeWidth="0.5" />
        {text}
      </svg>
      <h2>{props.data.source_id}</h2>
    </div>
  );
}

// MENU COMPONENT
function Menu() {
  const [allplasmids, setpls] = useState([]);
  const getAllPlasmids = () => {
    axios
      .get("http://localhost:5000/get_all")
      .then((response) => {
        var pls = response.data;
        // add a output_to column, that contains a list.
        // This list will be filled with only one "next" for the plasmids of type "L1"
        // For the L2 type, the list is all the "slot_*" columns that are not empty.
        var plasmids = pls.map((plasmid) => {
          var output_to = [];
          if (plasmid.type == "L1") {
            output_to.push("L1");
          } else {
            for (var i = 1; i <= 6; i++) {
              if (plasmid["slot_" + i] != "") {
                output_to.push(plasmid["slot_" + i]);
              }
            }
          }
          plasmid.output_to = output_to;
          return plasmid;
        });

        setpls(plasmids);

        console.log(response.data);
      })
      .catch((error) => {
        console.error("There was an error!", error);
      });
  };
  useEffect(getAllPlasmids, []);

  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div id="menu">
      {allplasmids.map((plasmid) => (
        <div
          className="draggable_plasmid"
          onDragStart={(event) => onDragStart(event, "default")}
          key={plasmid.source_id}
          draggable
        >
          <Plasmid data={plasmid} />
        </div>
      ))}
    </div>
  );
}

const initialNodes = [
  {
    id: "1",
    type: "input",
    data: { label: "input node" },
    position: { x: 250, y: 5 },
  },
];

let id = 0;
const getId = () => `dndnode_${id++}`;

function AppComponent() {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), []);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");

      // check if the dropped element is valid
      if (typeof type === "undefined" || !type) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });
      const newNode = {
        id: getId(),
        type,
        position,
        data: { label: `${type} node` },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance]
  );

  return (
    <div className="App">
      <ReactFlowProvider>
        <div className="reactflow-wrapper" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            fitView
          >
            <Controls />
          </ReactFlow>
        </div>
        <Menu />
      </ReactFlowProvider>
    </div>
  );
}

export default AppComponent;
