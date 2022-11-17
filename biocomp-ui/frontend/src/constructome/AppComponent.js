import React from "react";
import axios from "axios";
import { useState, useEffect, useRef, useCallback } from "react";
import Util from "../util.jsx";
import "./style.css";
import ReactFlow, { ReactFlowProvider } from "reactflow";
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
import Fuse from "fuse.js";

import ComputeComponent from "../ComputeComponent.jsx";
import "../style.css";

function RecursiveDict({ dict }) {
  if (dict === null || dict === undefined) {
    return;
  }

  return (
    <ul className={Array.isArray(dict) ? "array" : "dict"}>
      {Object.keys(dict).map((key) => {
        if (
          key === "samples" ||
          dict[key] === null ||
          dict[key] === undefined ||
          dict[key] === "" ||
          dict[key].length === 0 ||
          (typeof dict[key] === "object" && Object.keys(dict[key]).length === 0)
        ) {
          return;
        }
        if (typeof dict[key] === "boolean") {
          return (
            <li key={key}>
              <b>{key}</b>: {dict[key].toString()}
            </li>
          );
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
  // data status can be either "ok", "missing" or "partial"
  // ok means all the xp.samples.has_data are true
  // missing means all the xp.samples.has_data are false
  // partial means some are true and some are false
  var hasdata = 0;
  xp.data.samples.forEach((sample) => {
    if (sample.has_data) {
      hasdata += 1;
    }
  });
  var datastatus = "ok";
  if (hasdata === 0) {
    datastatus = "missing";
  } else if (hasdata < xp.data.samples.length) {
    datastatus = "partial";
  }

  return (
    <li className={"selectable expandable " + sel} onClick={() => handleClick(xp.data.name)}>
      <Flex>
        <h2>{xp.data.name}</h2> <span className="tag date">{xp.data.flow_date}</span>
        <span className={"tag data_status " + datastatus}>{datastatus}</span>
      </Flex>

      <div className="expandableContent">
        <RecursiveDict dict={xp.data} />
      </div>
    </li>
  );
}

function TU({ name, parts }) {
  return (
    <div className="parts">
      {parts.map((part) => {
        return <h4 key={part}>{part}</h4>;
      })}
    </div>
  );
}

function Plasmid({ name, tus, ratio }) {
  const l1orl2 = tus.length === 1 ? "l1" : "l2";

  var l2 = l1orl2 === "l2" ? <h3 className="l2">{name}</h3> : "";

  var tucontent = (
    <ul>
      {tus.map((tu) => {
        return (
          <li key={tu.TU}>
            <h3 className="l1">{name}</h3>
            <TU name={tu.TU} parts={tu.parts} />
          </li>
        );
      })}
    </ul>
  );

  const ratioelmt = ratio ? <span className="tag ratio">{ratio.toFixed(2)}</span> : null;

  return (
    <div className={"plasmid"}>
      {ratioelmt}
      {l2}
      {tucontent}
    </div>
  );
}

function RecipeRow({ recipe, handleClick }) {
  const sel = recipe.selected ? "selected" : "";
  return (
    <li
      className={"expandable selectable " + sel}
      onClick={() => handleClick(recipe.data.name)}
      key={recipe.data.name}
    >
      <h2>{recipe.data.name}</h2>
      <div className="expandableContent recipe">
        <Text className="recipe_description">{recipe.data.description}</Text>
        {recipe.data.aggregations.map((agg) => {
          return (
            <div className="aggregation" key={agg.name}>
              {agg.sources.map((cotx) => {
                return (
                  <Plasmid
                    name={cotx.source}
                    tus={cotx.tus}
                    ratio={agg.sources.length > 1 ? cotx.ratio : null}
                    key={cotx.source}
                  />
                );
              })}
            </div>
          );
        })}
      </div>
    </li>
  );
}

function Graph({ graph }) {
	if (graph === null || graph === undefined) {
		return;
	}
	return(
          <ReactFlowProvider>
            <ComputeComponent data={graph} />
          </ReactFlowProvider>
	);
}

function AppComponent() {
  const [xpList, setXpList] = useState([]);
  const [recipeList, setRecipeList] = useState([]);
  const [graph, setGraph] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:4321/xps").then((response) => {
      response.data.forEach((xp) => {
        xp.selected = false;
        xp.keep = true;
      });
      setXpList(response.data);
    });
    axios.get("http://localhost:4321/recipes").then((response) => {
      response.data.forEach((recipe) => {
        recipe.selected = false;
        recipe.keep = true;
      });
      setRecipeList(response.data);
    });
  }, []);

  const filterRecipes = (name) => {
    const new_list = recipeList.map((item) => {
      item.keep = name ? item.data.xps.includes(name) : true;
      return item;
    });
    setRecipeList(new_list);
  };

  const filterXps = (name) => {
    const new_list = xpList.map((item) => {
      const reclist = item.data.samples.map((sample) => sample.recipe);
      item.keep = name ? reclist.includes(name) : true;
      return item;
    });
    setXpList(new_list);
  };

  const selectItem = (name, list, setlist, filterOther) => {
    const new_list = list.map((item) => {
      if (item.data.name === name) {
        filterOther(item.selected ? null : name);
        item.selected = !item.selected;
      } else {
        item.selected = false;
      }
      return item;
    });
    setlist(new_list);
  };

  const selectXp = (name) => {
    selectItem(name, xpList, setXpList, filterRecipes);
  };
  const selectRecipe = (name) => {
    selectItem(name, recipeList, setRecipeList, filterXps);
      setGraph(null);
    axios.get("http://localhost:4321/network/" + name).then((response) => {
      console.log(response.data);
      setGraph(response.data);
      console.log("graph", graph);
    });
  };

  return (
    <Flex direction="column" h="100vh">
      <Box w="100%" className="header">
        <h1>The Constructome Browser</h1>
      </Box>
      <Flex>
        <Box id="xplist" className="mainlist">
          {xpList.map((xp) =>
            xp.keep ? <XpRow xp={xp} handleClick={selectXp} key={xp.data.name} /> : null
          )}
        </Box>
        <Box id="recipelist" className="mainlist">
          {recipeList.map((recipe) =>
            recipe.keep ? (
              <RecipeRow recipe={recipe} handleClick={selectRecipe} key={recipe.data.name} />
            ) : null
          )}
        </Box>
        <Box id="graph" className="mainlist">
			<Graph graph={graph} />
        </Box>
      </Flex>
    </Flex>
  );
}

export default AppComponent;
