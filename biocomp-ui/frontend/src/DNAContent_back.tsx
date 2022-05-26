import React, { ReactNode } from "react"

function DNAContent(props: any) {
  const content = props.data.content.map((c: string, i: number) => (
    <li>
      <img src={require("./grnsymbols/" + props.data.content_type[i] + ".png")} alt={c} />
      <span className="dna-elmt">{c}</span>
    </li>
  ))
  return (
    <div className="dna-content">
      {content}

      <li>
        <img src={require("./grnsymbols/terminator.png")} alt="terminator symbol" />
      </li>

      <div className="dna-name"></div>
    </div>
  )
}
export default DNAContent
