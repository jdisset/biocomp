import React from "react";
import axios from "axios";
import { useState, useEffect, useRef, useCallback } from "react";
import Util from "../util.jsx";
import "./style.css";
import {
  Flex,
  Spacer,
  Center,
  Box,
  Square,
  Text,
  Heading,
  Stack,
  StackDivider,
  Card,
  CardHeader,
  CardBody,
  CardFooter,
} from "@chakra-ui/react";

function RecursiveDict({ dict }) {
  if (dict === null || dict === undefined) {
    return;
  }

  return (
    <ul className={Array.isArray(dict) ? "array" : "dict"}>
      {Object.keys(dict).map((key) => {
        if (
          dict[key] === null ||
          dict[key] === undefined ||
          dict[key] === "" ||
          dict[key].length === 0 ||
		  (typeof dict[key] === "object" && Object.keys(dict[key]).length === 0)

        ) {
          return;
        }
        if (typeof dict[key] === "object") {
          if (Array.isArray(dict)) {
            return (
              <li key={key}>
                <RecursiveDict dict={dict[key]} />
              </li>
            );
          }
          return (
            <li key={key}>
              <h3>{key}</h3> <RecursiveDict dict={dict[key]} />
            </li>
          );
        } else {
          return (
            <li key={key}>
              <h3>{key}</h3> {dict[key]}
            </li>
          );
        }
      })}
    </ul>
  );
}

function XpRow({ xp, handleClick }) {
  const sel = xp.selected ? "selected" : "";
  return (
    // check if selected is true:
    // if it is, add the class "selected":
    <li className={"selectable expandable " + sel} onClick={() => handleClick(xp.name)}>
      <h2>{xp.name}</h2>
      <div className="expandableContent">
        <RecursiveDict dict={xp} />
      </div>
    </li>
  );
}

function RecipeRow({ recipe }) {
  return (
    <li className={"expandable selectable"}>
      <h2>{recipe.name}</h2>
      <div className="expandableContent">
        <RecursiveDict dict={recipe} />
      </div>
    </li>
  );
}

function AppComponent() {
  // we need to make an axios request to the server to get the list of xp:
  const [xpList, setXpList] = useState([]);
  // and for recipes:
  const [recipeList, setRecipeList] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:4321/xp").then((response) => {
      setXpList(response.data);
    });
    axios.get("http://localhost:4321/recipe").then((response) => {
      setRecipeList(response.data);
    });
  }, []);

  const selectXp = (name) => {
    console.log("selected xp: ", name);
    // then we set slected to true for the xp with the name:
    // and we set selected to false for all other xps:
    const new_xp_list = xpList.map((xp) => {
      xp.selected = xp.name === name;
      return xp;
    });
    setXpList(new_xp_list);
  };

  const selectXpCallback = useCallback(selectXp, [xpList]);

  return (
    <Flex direction="column" h="100vh">
      <Box w="100%" className="header">
        <h1>The Constructome</h1>
      </Box>
      <Flex>
        <Box id="xplist" className="mainlist">
          {xpList.map((xp) => (
            // we want to be able to select an xp by clicking on it:
            <XpRow xp={xp} handleClick={selectXpCallback} />
          ))}
        </Box>
        <Box id="recipelist" className="mainlist">
          {recipeList.map((recipe) => (
            <RecipeRow recipe={recipe} />
          ))}
        </Box>
      </Flex>
    </Flex>
  );
}

export default AppComponent;
