import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from "streamlit-component-lib"
import SEQNode from "./SEQNode.jsx"
import TLNode from "./TLNode.jsx"
import TCNode from "./TCNode.jsx"
import INNode from "./INNode.jsx"
import OUTNode from "./OUTNode.jsx"
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
  translation: TLNode,
  transcription: TCNode,
  constant: INNode,
  out: OUTNode,
}

function ComputeComponent(props) {
  const styled_edges = props.data.edges.map((e) => ({ style: { stroke: "black", strokeWidth: "0.5" }, ...e, }))
  const layouted = Util.getLayoutedElements(props.data.nodes, styled_edges, 140, 100)
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

export default ComputeComponent
