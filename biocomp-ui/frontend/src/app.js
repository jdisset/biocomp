import "regenerator-runtime/runtime";
import React from "react";
import ReactDOM from "react-dom/client";
import AppComponent from "./constructome/AppComponent";
import "./style.css";
import { ChakraProvider } from '@chakra-ui/react'

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <ChakraProvider>
      <AppComponent />
    </ChakraProvider>
  </React.StrictMode>
);
