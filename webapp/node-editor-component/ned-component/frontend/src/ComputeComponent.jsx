import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from "streamlit-component-lib"
import SEQNode from "./SEQNode.jsx"
import TLNode from "./TLNode.jsx"
import TCNode from "./TCNode.jsx"
import INNode from "./INNode.jsx"
import OUTNode from "./OUTNode.jsx"
import CTENode from "./CTENode.jsx"
import React, { ReactNode } from "react"
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer"

import Util from "./util.jsx"

const nodeTypes = {
  sequestron_ERN: SEQNode,
  sequestron_RECOMBINASE: SEQNode,
  translation: TLNode,
  transcription: TCNode,
  bias: CTENode,
  in: INNode,
  out: OUTNode,
}

const typeDim = {
  sequestron_ERN: { width: 100, height: 50 },
  sequestron_RECOMBINASE: { width: 100, height: 50 },
  translation: { width: 30, height: 50 },
  transcription: { width: 30, height: 50 },
  bias: { width: 45, height: 63 },
  output: { width: 20, height: 10 },
}

function hasEdgeLabel(data) {
  if (data.tgtdata.parameters) {
    if (data.tgtdata.type === "transcription" || data.tgtdata.type === "translation") {
      return true
    }
  }
  return false
}

function getRate(data) {
  if (hasEdgeLabel(data)) {
    var i = parseInt(data.tgthandle)
    return data.tgtdata.parameters.tr_rates[i]
  } else return 0
}
function getEdgeLabel(data) {
  if (hasEdgeLabel(data)) {
    return getRate(data).toFixed(2)
  }
  return ""
}

function ComputeComponent(props) {
  const styled_edges = props.data.edges.map((e) => ({
    style: {
      stroke: hasEdgeLabel(e.data) ? Util.cmap(getRate(e.data)) : "black",
      strokeWidth: 0.5 + getRate(e.data)*2.0,
    },
    label: getEdgeLabel(e.data),
    ...e,
  }))
  const layouted = Util.getLayoutedElements(props.data.nodes, styled_edges, 60, 60, typeDim)
  const [nodes, setNodes, onNodesChange] = useNodesState(layouted.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges)

  return (
    <div style={{ width: "100%", height: 1000 }}>
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

export default ComputeComponent
