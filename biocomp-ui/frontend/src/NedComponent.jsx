import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";

import ReactTooltip from "react-tooltip";
import DNAContent from "./DNAContent.jsx";
import GRNComponent from "./GRNComponent.jsx";
import ComputeComponent from "./ComputeComponent.jsx";
import React, { ReactNode } from "react";

class NedComponent extends StreamlitComponentBase {
  render() {
    console.log(this.props);
    switch (this.props.args["output_type"]) {
      case "GRN":
        return <GRNComponent data={this.props.args} />;
      case "COMPUTE":
        return <ComputeComponent data={this.props.args} />;
      case "DNA":
        let initexpanded = this.props.args.initexpanded | false;
        const dnaData = this.props.args.nodes.map((d) =>
          d.type === "DNA" ? <DNAContent data={d.data} initExpanded={initexpanded} /> : ""
        );
        return <div className="dna-list">{dnaData}</div>;
      default:
        return <em>ERROR: Unknown output type</em>;
    }
  }
}

const StreamlitComponent = withStreamlitConnection(NedComponent);
const StaticComponent = NedComponent;

export default StreamlitComponent;

export { StaticComponent };
