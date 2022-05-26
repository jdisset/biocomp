import "regenerator-runtime/runtime";
import React from "react";
import ReactDOM from "react-dom/client";
import { StaticComponent } from "./NedComponent";
import "./style.css";

const windowUrl = window.location.search;
const params = new URLSearchParams(windowUrl);
const root = ReactDOM.createRoot(document.getElementById("root"));
let args = JSON.parse(params.get("args"));
//root.render(<div> {JSON.stringify(args)} </div>);
root.render(
  <React.StrictMode>
    <StaticComponent args={args} />
  </React.StrictMode>
);
