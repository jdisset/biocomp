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
import React, { ReactNode, useRef, useEffect, useMemo, useCallback } from "react";

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

// there can be a handleNodeChange functions passed
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

  const styled_nodes = props.data.nodes.map((n) => ({
    ...n,
    type: n.type == "input" ? "in" : n.type == "output" ? "out" : n.type,
  }));

  const dagreGraph = new dagre.graphlib.Graph({ multigraph: true, compound: true });
  const layouted = Util.getLayoutedElements(
    dagreGraph,
    styled_nodes,
    styled_edges,
    30,
    80,
    typeDim,
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layouted.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges);





  



  useEffect(() => {
    if (props.handleNodeChange) {
      const new_nodes = props.handleNodeChange(nodes);
      // check if their json representation is the same
      if (new_nodes != undefined && JSON.stringify(new_nodes) != JSON.stringify(nodes)) {
        setNodes(new_nodes);
      }
      console.log("nodes changed");
    }
  }, [nodes]);

  function fillEdgesWithNodeData(edges, nodes) {
    return edges.map((e) => {
      const src_id = parseInt(e.data.source_node_list_id);
      const tgt_id = parseInt(e.data.target_node_list_id);
      const src_node = nodes[src_id];
      const tgt_node = nodes[tgt_id];
      return {
        ...e,
        data: {
          ...e.data,
          srcdata: src_node.data,
          tgtdata: tgt_node.data,
        },
      };
    });
  }

  return (
    <div
      style={{
        width: props.data.width === undefined ? "100%" : props.data.width,
        height: props.data.height === undefined ? 1000 : props.data.height,
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={fillEdgesWithNodeData(edges, nodes)}
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
