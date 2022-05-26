import "regenerator-runtime/runtime";
import React from "react";
import ReactDOM from "react-dom/client";
import StreamlitComponent from "./NedComponent";
import "./style.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <StreamlitComponent/>
  </React.StrictMode>
);
