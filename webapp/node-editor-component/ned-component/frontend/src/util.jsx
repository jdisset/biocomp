import dagre from "dagre"

let colormap = require("colormap")
const NCOLORS = 100
const colors = colormap({
  colormap: [
    { index: 0.0, rgb: [41, 41, 41] },
    { index: 0.33, rgb: [86, 34, 50] },
    { index: 0.67, rgb: [173, 42, 92] },
    { index: 1.0, rgb: [233, 43, 71] },
  ],
  nshades: NCOLORS,
  format: "hex",
  alpha: 1,
})

class Util {
  static cmap(x) {
    console.log(colors)
    return colors[Math.max(0, Math.min(NCOLORS - 1, Math.floor(x * (NCOLORS - 1))))]
  }
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
