import ContentEdge from "./ContentEdge.jsx";
import dagre from "dagre";
import React, { ReactNode, useRef, useEffect, useMemo, useCallback } from "react";
import ReactFlow, { useNodesState, useEdgesState, useNodesInitialized } from "reactflow";

import Util from "./util.jsx";
import { typeDim, computeNodeTypes } from "./util.jsx";

import "reactflow/dist/style.css";
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

  const nodesInitialized = useNodesInitialized({});

  useEffect(() => {
    if (props.nodeInitHook) {
      props.nodeInitHook(nodesInitialized);
    }
  }, [nodesInitialized]);

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
    <div id="graph" style={{ height: "100vh", width: "100vw" }}>
      <ReactFlow
        nodes={nodes}
        edges={fillEdgesWithNodeData(edges, nodes)}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={computeNodeTypes}
        edgeTypes={computeEdgeTypes}
      />
    </div>
  );
}

export default ComputeComponent;
