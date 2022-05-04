import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from "streamlit-component-lib"
import DNANode from "./DNANode"
import RNANode from "./RNANode.jsx"
import PRTNode from "./PRTNode.jsx"
import React, { ReactNode } from "react"
import Util from "./util.jsx"
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer"

const nodeTypes = { DNA: DNANode, RNA: RNANode, PRT: PRTNode }

const typeDim = {
  DNA: { width: 180, height: 350 },
  RNA: { width: 180, height: 100 },
  PRT: { width: 180, height: 100 },
}

function GRNComponent(props) {
  const styled_edges = props.data.edges.map((e) => ({
    style: { stroke: "black", strokeWidth: "0.5" },
    ...e,
  }))
  const layouted = Util.getLayoutedElements(props.data.nodes, styled_edges, 60, 60, typeDim)
  const [nodes, setNodes, onNodesChange] = useNodesState(layouted.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges)

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
  )
}

export default GRNComponent
