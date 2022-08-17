import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import SEQNode from "./SEQNode.jsx";
import TLNode from "./TLNode.jsx";
import TCNode from "./TCNode.jsx";
import INNode from "./INNode.jsx";
import OUTNode from "./OUTNode.jsx";
import CTENode from "./CTENode.jsx";
import ContentEdge from "./ContentEdge.jsx";

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

const nodeTypes = {
  sequestron_ERN: SEQNode,
  sequestron_RECOMBINASE: SEQNode,
  translation: TLNode,
  transcription: TCNode,
  bias: CTENode,
  in: INNode,
  out: OUTNode,
};

const edgeTypes = {
  content: ContentEdge,
};

const typeDim = {
  sequestron_ERN: { width: 100, height: 70 },
  sequestron_RECOMBINASE: { width: 100, height: 70 },
  translation: { width: 40, height: 80 },
  transcription: { width: 40, height: 80 },
  bias: { width: 45, height: 40 },
  output: { width: 20, height: 20 },
};

function getEdgeLabel(data) {
  return data.srccdg.content.join(", ");
}

function ComputeComponent(props) {
  const exportRef = React.createRef();
  const rootRef = useRef(null);
  const onClick = () => {
    const elements = rootRef.current;
    Util.exportAsImage(elements, "test");
  };
  const styled_edges = props.data.edges.map((e) => ({
    style: {
      //stroke: hasEdgeLabel(e.data) ? Util.cmap(getRate(e.data)) : "black",
      stroke: "black",
      strokeWidth: 0.5,
      //strokeWidth: 0.5 + Math.max(getRate(e.data) * 4.0, 0),
    },
    //label: getEdgeLabel(e.data),
    label: "test",
    type: "content",
    ...e,
  }));
  const layouted = Util.getLayoutedElements(props.data.nodes, styled_edges, 60, 60, typeDim);
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
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
      />
    </div>
  );
}

export default ComputeComponent;
