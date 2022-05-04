import dagre from "dagre"

class Util {
  static getLayoutedElements = (
    nodes,
    edges,
    nodeWidth = 150,
    nodeHeight = 270,
    dimensionsDict = {},
    direction = "TB"
  ) => {
    const dagreGraph = new dagre.graphlib.Graph()
    dagreGraph.setDefaultEdgeLabel(() => ({}))
    const isHorizontal = direction === "LR"
    dagreGraph.setGraph({ rankdir: direction })
    nodes.forEach((node) => {
      const w = node.type in dimensionsDict ? dimensionsDict[node.type].width : nodeWidth
      const h = node.type in dimensionsDict ? dimensionsDict[node.type].height : nodeHeight
      dagreGraph.setNode(node.id, { width: w, height: h })
    })
    edges.forEach((edge) => {
      dagreGraph.setEdge(edge.source, edge.target)
    })
    dagre.layout(dagreGraph)
    nodes.forEach((node) => {
      const w = node.type in dimensionsDict ? dimensionsDict[node.type].width : nodeWidth
      const h = node.type in dimensionsDict ? dimensionsDict[node.type].height : nodeHeight
      const nodeWithPosition = dagreGraph.node(node.id)
      node.targetPosition = isHorizontal ? "left" : "top"
      node.sourcePosition = isHorizontal ? "right" : "bottom"
      // We are shifting the dagre node position (anchor=center center) to the top left
      // so it matches the React Flow node anchor point (top left).
      node.position = {
        x: nodeWithPosition.x - w / 2,
        y: nodeWithPosition.y - h / 2,
      }
      return node
    })
    return { nodes, edges }
  }
}
export default Util
