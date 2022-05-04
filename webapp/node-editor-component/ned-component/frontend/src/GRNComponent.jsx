import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from "streamlit-component-lib"
import DNANode from "./DNANode"
import RNANode from "./RNANode.jsx"
import PRTNode from "./PRTNode.jsx"
import React, { ReactNode } from "react"
import dagre from "dagre"
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer"

const dagreGraph = new dagre.graphlib.Graph()
dagreGraph.setDefaultEdgeLabel(() => ({}))

const nodeWidth = 150
const nodeHeight = 270

const getLayoutedElements = (nodes, edges, direction = "TB") => {
  const isHorizontal = direction === "LR"
  dagreGraph.setGraph({ rankdir: direction })
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight })
  })
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target)
  })
  dagre.layout(dagreGraph)
  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id)
    node.targetPosition = isHorizontal ? "left" : "top"
    node.sourcePosition = isHorizontal ? "right" : "bottom"
    // We are shifting the dagre node position (anchor=center center) to the top left
    // so it matches the React Flow node anchor point (top left).
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    }
    return node
  })
  return { nodes, edges }
}

const nodeTypes = { DNA: DNANode, RNA: RNANode, PRT: PRTNode }

function GRNComponent(props) {
  const styled_edges = props.data.edges.map((e) => ({
    style: { stroke: "black", "strokeWidth": "0.5" },
    ...e,
  }))
  const layouted = getLayoutedElements(props.data.nodes, styled_edges)
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
