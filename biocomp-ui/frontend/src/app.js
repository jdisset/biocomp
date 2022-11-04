import "regenerator-runtime/runtime";
import React from "react";
import ReactDOM from "react-dom/client";
import AppComponent from './AppComponent';
import "./style.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <AppComponent />
  </React.StrictMode>
);
