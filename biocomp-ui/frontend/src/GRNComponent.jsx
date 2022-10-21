import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import DNANode from "./DNANode.jsx";
import RNANode from "./RNANode.jsx";
import PRTNode from "./PRTNode.jsx";
import React, { ReactNode } from "react";
import Util from "./util.jsx";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer";

const nodeTypes = { DNA: DNANode, RNA: RNANode, PRT: PRTNode };

const typeDim = {
  DNA: { width: 180, height: 350 },
  RNA: { width: 180, height: 100 },
  PRT: { width: 180, height: 100 },
};

function GRNComponent(props) {
  console.log("GRNComponent");
  const styled_edges = props.data.edges.map((e) => ({
    style: { stroke: "black", strokeWidth: "0.5" },
    ...e,
  }));
	console.log("hello 0");
  const layouted = Util.getLayoutedElements(props.data.nodes, styled_edges, 60, 60, typeDim);
	console.log("hello 1");
  const [nodes, setNodes, onNodesChange] = useNodesState(layouted.nodes);
	console.log("hello 2");
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges);

  console.log("nodes", nodes);
  console.log("edges", edges);

  return (
    <div style={{ width: "100%", height: 800 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
      />
    </div>
  );
}

export default GRNComponent;
