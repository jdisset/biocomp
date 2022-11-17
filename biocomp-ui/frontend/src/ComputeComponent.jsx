import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import SEQNode from "./SEQNode.jsx";
import AGGNode from "./AGGNode.jsx";
import SRCNode from "./SRCNode.jsx";
import TLNode from "./TLNode.jsx";
import TCNode from "./TCNode.jsx";
import INNode from "./INNode.jsx";
import OUTNode from "./OUTNode.jsx";
import NUMNode from "./NUMNode.jsx";
import CTENode from "./CTENode.jsx";
import INVNode from "./INVNode.jsx";
import DENDNode from "./DENDNode.jsx";
import ContentEdge from "./ContentEdge.jsx";
import dagre from "dagre";

import images from "./grnsymbols/*.png";
import React, { ReactNode, useRef } from "react";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer";

import html2canvas from "html2canvas";
import Util from "./util.jsx";
// from util we also want typeDim
import { typeDim, computeNodeTypes } from "./util.jsx";


console.log("ComputeComponent.jsx loaded");

function getEdgeLabel(data) {
  return data.srccdg ? data.srccdg.content.join(", ") : "";
}

const computeEdgeTypes = {
  content: ContentEdge,
};

function ComputeComponent(props) {
  if (!props.data) {
    return null;
  }
  const exportRef = React.createRef();
  const rootRef = useRef(null);
  const onClick = () => {
    const elements = rootRef.current;
    Util.exportAsImage(elements, "test");
  };
  const styled_edges = props.data.edges.map((e) => ({
    style: {
      stroke: "black",
      strokeWidth: 0.5,
    },
    label: getEdgeLabel(e.data),
    type: "content",
    ...e,
  }));

  const dagreGraph = new dagre.graphlib.Graph();
  const layouted = Util.getLayoutedElements(dagreGraph, props.data.nodes, styled_edges, 60, 60, typeDim);
  const [nodes, setNodes, onNodesChange] = useNodesState(layouted.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges);
  return (
    <div
      style={{
        width: props.data.width === undefined ? "100%" : props.data.width,
        height: props.data.height === undefined ? 1000 : props.data.height,
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={computeNodeTypes}
        edgeTypes={computeEdgeTypes}
        fitView
      />
    </div>
  );
}

export default ComputeComponent;
