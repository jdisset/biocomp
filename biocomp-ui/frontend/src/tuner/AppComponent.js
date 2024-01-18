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
} from "reactflow";

/*════════════════════════════════════════════════════════════════════════════════*/

const NETWORK_NAME = "bp_v1";

function GraphWithResults() {
  const reactFlowInstance = useReactFlow();

  const [graph, setGraph] = useState(null);
  const [params, setParams] = useState(null);

  const [paramsPassedToNodes, setParamsPassedToNodes] = useState(false);


  // PARAMETERS
  const changeNodeParams = useCallback((id, pdata) => {
    const newParams = { [id]: pdata };
    //console.log(`Received new params for node ${id}:`, newParams);
    setParams((prev) => {
      return { ...prev, ...newParams };
    });
  }, []);

  const [nodesInitialized, setNodesInitialized] = useState(false);

  const afterNodeInit = useCallback((status) => {
    setNodesInitialized(status);
  }, []);

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const dataParam = searchParams.get('init_params');
    console.log("dataParam", dataParam);
    console.log("Fetching graph and params for network", NETWORK_NAME);
    axios.get(`http://localhost:4321/network/${NETWORK_NAME}`, { params: { init_params: dataParam} }).then((response) => {
      setGraph(response.data);
    });
    axios.get(`http://localhost:4321/params/${NETWORK_NAME}`).then((response) => {
      setParams(response.data);
    });
  }, []);

  useEffect(() => {
    if (params != null && !paramsPassedToNodes) {
      const nodes = reactFlowInstance.getNodes();
      const new_nodes = nodes.map((node) => {
        if (node.id in params) {
          node.data = {
            ...node.data,
            tunable: params[node.id],
            updateMyParams: (pdata) => {
                changeNodeParams(node.id, pdata);
            },
          };
        }
        return node;
      });
      reactFlowInstance.setNodes(new_nodes);
      setParamsPassedToNodes(true);
    }
  }, [paramsPassedToNodes, params, nodesInitialized]);

  // SIMULATION

  const [simdata, setSimdata] = useState("");
  useEffect(() => {
    const args = { params: params, network_name: NETWORK_NAME };
    //console.log(`Simulation params:`, encodeURIComponent(JSON.stringify(params)));
    //as base64:
    console.log(`Simulation params:`, btoa(JSON.stringify(params)));
    axios.post("http://localhost:4321/simulate", args).then((response) => {
      setSimdata(response.data.image);
      console.log("Received simulation results");
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
      <ComputeComponent data={graph} nodeInitHook={afterNodeInit} />
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

function AppComponent() {
  return (
    <ReactFlowProvider>
      <GraphWithResults />
    </ReactFlowProvider>
  );
}

export default AppComponent;
