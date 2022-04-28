import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from "streamlit-component-lib"

import ReactTooltip from "react-tooltip"
import DNAContent from "./DNAContent.jsx"
import GRNComponent from "./GRNComponent.jsx"
import React, { ReactNode } from "react"

interface Point {
  x: number
  y: number
}
interface Node {
  id: string
  position: Point
}
interface Edge {
  id: string
  source: string
  target: any
}
interface State {
  nodes: any
  edges: any
}

class NedComponent extends StreamlitComponentBase {
  public render = (): ReactNode => {
    switch (this.props.args["output_type"]) {
      case "GRN":
        return <GRNComponent data={this.props.args} />
      case "compute":
        return <em>Compute Graph</em>
      case "DNA":
        const dnaData = this.props.args.nodes.map((d: any) =>
          d.type === "dna" ? <DNAContent data={d.data} /> : ""
        )
        return <div className="dna-list">{dnaData}</div>
      default:
        return <em>ERROR: Unknown output type</em>
    }
  }
}

export default withStreamlitConnection(NedComponent)
