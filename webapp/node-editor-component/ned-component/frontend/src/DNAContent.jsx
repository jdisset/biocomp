import React, { ReactNode } from "react"

const zeroPad = (num, places) => String(num).padStart(places, "0")

function DNAContent({ initExpanded = false, expandable = true, ...props }) {
  const content = props.data.content.map((c, i) => (
    <li>
      <img src={require("./grnsymbols/" + props.data.content_type[i] + ".png")} alt={c} />
      <span className="dna-elmt">{c}</span>
    </li>
  ))
  const [expanded, setExpanded] = React.useState(initExpanded)
  return (
    <div
      className={"dna-content" + (expanded ? " expanded" : " notexpanded")}
      onMouseEnter={() => setExpanded(expandable ? true : initExpanded)}
      onMouseLeave={() => setExpanded(expandable ? false : initExpanded)}
    >
      {content}
      <li>
        <img src={require("./grnsymbols/terminator.png")} alt="terminator symbol" />
      </li>

      <div className="dna-name">DNA {zeroPad(parseInt(props.data.id), 2)}</div>
    </div>
  )
}
export default DNAContent
