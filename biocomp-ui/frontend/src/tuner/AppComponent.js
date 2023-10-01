/*──────────────────────────────▼     imports     ▼───────────────────────────────*/

import axios from "axios";
import React, { useState, useEffect, useRef, useCallback } from "react";
import Util from "../util.jsx";
import ComputeComponent from "../ComputeComponent.jsx";
import "../style.css";
import styled from "styled-components";

import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  useNodes,
} from "react-flow-renderer";

/*════════════════════════════════════════════════════════════════════════════════*/

function AppComponent() {
  const [graph, setGraph] = useState(null);
  const [params, setParams] = useState(null);

  const changeNodeParams = useCallback((id, pdata) => {
    const newParams = { id: pdata };
    print("changeNodeParams", id, newParams);
    setParams((prev) => {
      return { ...prev, ...newParams };
    });
  }, []);

  const handleNodeChange = (nodes) => {
    // for each param
    if (params == null) {
      return nodes;
    }
    console.log("handleNodeChange");

    let new_params = {};

    const new_nodes = nodes.map((node) => {
      if (node.id in params) {
        for (const [path, i, name, value] of params[node.id]) {
          // add a params object to the node
          if (!("tunable" in node.data)) {
            node.data["tunable"] = [];
            node.data["tunable"].push([path, i, name, value]);
            node.data["updateMyParams"] = (pdata) => {
              changeNodeParams(node.id, pdata);
            };
          }
          // update the params object
          else {
            console.assert(node.data["tunable"].length == params[node.id].length);
            // then we update the params object
            new_params[node.id] = node.data["tunable"];
          }
        }
      }
      return node;
    });

    return new_nodes;
  };

  useEffect(() => {
    axios.get("http://localhost:4321/network").then((response) => {
      setGraph(response.data);
    });
    axios.get("http://localhost:4321/params").then((response) => {
      setParams(response.data);
    });
  }, []);

  const [simdata, setSimdata] = useState('');

  useEffect(() => {
    const args = { params: params };
    console.log("sending", args);
    axios.post("http://localhost:4321/simulate", args).then((response) => {
      // response contains image, a base64 encoded png
      setSimdata(response.data.image);
    });
  }, [params]);

  // make the image blink when it changes
  useEffect(() => {
    const img = document.getElementById("results").firstChild;
    img.style.opacity = 0.5;
    setTimeout(() => {
      img.style.opacity = 1.0;
    }, 100);
  }, [simdata]);



  return (
    <>
      <div id="graph" className="graph">
        <ReactFlowProvider>
          <ComputeComponent data={graph} handleNodeChange={handleNodeChange} />
        </ReactFlowProvider>
      </div>
      <div id="results">
        <img
          src={simdata}
          style={{ position: "absolute", top: "0px", left: "0px", backgroundColor: "grey" }}
          width="400px"
          height="400px"
        />
      </div>
    </>
  );
}

export default AppComponent;
