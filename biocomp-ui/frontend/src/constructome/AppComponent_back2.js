/*──────────────────────────────▼     imports     ▼───────────────────────────────*/

import axios from "axios";
import React, { useState, useEffect, useRef, useCallback } from "react";
import Util from "../util.jsx";
import "./style.css";
import ReactFlow, { ReactFlowProvider } from "reactflow";
import Fuse from "fuse.js";
import RecursiveDict from "./RecursiveDictComponent.jsx";
import ComputeComponent from "../ComputeComponent.jsx";
import "../style.css";
import styled from "styled-components";

/*════════════════════════════════════════════════════════════════════════════════*/

/*───────────────────────▼    base styled components     ▼────────────────────────*/

const Flex = styled.div`
  display: flex;
`;

const Main = styled.div`
  display: flex;
  flex-direction: column;
`;

/*════════════════════════════════════════════════════════════════════════════════*/

/*───────────────────────────────▼     XpRow     ▼────────────────────────────────*/

function XpRow({ xp, handleClick }) {
  const sel = xp.selected ? "selected" : "";
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

/*════════════════════════════════════════════════════════════════════════════════*/

/*─────────────────────────────▼     RecipeRow     ▼──────────────────────────────*/

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
        <p className="recipe_description">{recipe.data.description}</p>
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

/*════════════════════════════════════════════════════════════════════════════════*/

/*────────────────────────────▼     TU & Plasmid     ▼────────────────────────────*/

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

/*════════════════════════════════════════════════════════════════════════════════*/

/*───────────────────────────────▼     Graph     ▼────────────────────────────────*/

function Graph({ graph }) {
  if (graph === null || graph === undefined) {
    return;
  }
  return (
    <ReactFlowProvider>
      <ComputeComponent data={graph} />
    </ReactFlowProvider>
  );
}

/*════════════════════════════════════════════════════════════════════════════════*/

/*──────────────────────────────▼     OldApp     ▼────────────────────────────────*/

function OldAppComponent() {
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
      setGraph(response.data);
    });
  };

  return (
    <Main>
      <div className="header">
        <h1>The Constructome Browser</h1>
      </div>
      <Flex>
        <div id="xplist" className="mainlist">
          <h2 className="boxtitle">Experiments</h2>
          <div className="boxcontent">
            {xpList.map((xp) =>
              xp.keep ? <XpRow xp={xp} handleClick={selectXp} key={xp.data.name} /> : null
            )}
          </div>
        </div>
        <div id="recipelist" className="mainlist">
          <h2 className="boxtitle">Recipes</h2>
          <div className="boxcontent">
            {recipeList.map((recipe) =>
              recipe.keep ? (
                <RecipeRow recipe={recipe} handleClick={selectRecipe} key={recipe.data.name} />
              ) : null
            )}
          </div>
        </div>
        <div id="graph" className="mainlist">
          <Graph graph={graph} />
        </div>
      </Flex>
    </Main>
  );
}

/*════════════════════════════════════════════════════════════════════════════════*/

/*────────────────────────────────▼     App     ▼─────────────────────────────────*/

function NewAppComponent() {
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

  const selectXp = (name) => {
    selectItem(name, xpList, setXpList, filterRecipes);
  };
  const selectRecipe = (name) => {
    selectItem(name, recipeList, setRecipeList, filterXps);
    setGraph(null);
    axios.get("http://localhost:4321/network/" + name).then((response) => {
      setGraph(response.data);
    });
  };

  return (
    <Main>
      <div className="header">
        <h1>The Constructome Browser</h1>
      </div>
      <Flex>
        <div id="xplist" className="mainlist">
          <h2 className="boxtitle">Experiments</h2>
          <div className="boxcontent">
            {xpList.map((xp) =>
              xp.keep ? <XpRow xp={xp} handleClick={selectXp} key={xp.data.name} /> : null
            )}
          </div>
        </div>
        <div id="recipelist" className="mainlist">
          <h2 className="boxtitle">Recipes</h2>
          <div className="boxcontent">
            {recipeList.map((recipe) =>
              recipe.keep ? (
                <RecipeRow recipe={recipe} handleClick={selectRecipe} key={recipe.data.name} />
              ) : null
            )}
          </div>
        </div>
        <div id="graph" className="mainlist">
          <Graph graph={graph} />
        </div>
      </Flex>
    </Main>
  );
}

/*════════════════════════════════════════════════════════════════════════════════*/

export default NewAppComponent;

/*────────────────────────────────▼     TODO     ▼────────────────────────────────*/

//TODO
// We have a few highly hierarchical (nested) data structures:
// [ ] Xp
// [ ] Recipe
// [ ] L2
// [ ] L1
// [ ] TU/L0
// [ ] Part
//
// They should all match with a component, + maybe some wrappers such as:
// [ ] Plasmid
// [ ] Each of these components should have a short "tag" version and an expanded version.
// [ ] Color coding for each of these components types
//
//
// List system:
// All of these elements should be available to display in a list. By default, the XP and Recipe lists
// are shown and expanded. The other lists are just shown as, for example, "Parts (57)".
// When clicking on the list name, the list is expanded and a new column is dislayed
// (i.e a new list added to the list of displayed lists).
// [ ] List component with title and content (and filtering?)
//
// [ ] Each list has a setter state.
// [ ] We do everything client side for now, so we can just asl /xps and /recipes and /L2s etc. and get all the data.
// Later
//
//
// Search & filter:
// We want to be able to search for any of these elements, and filter any list to only show elements that contain them (including this element itself, as well).
// There will be a
// [ ] Search bar
// That will suggest in real time, as we type, a list of elements that match the search. Use fuse.js
// We can pin elements to the filter bar, and they will be displayed in a list of pinned elements that
// contributes to the filtering of the displayed lists.
// [ ] Filter bar
// [ ] Filter component (with option for: exact match, fuzzy, case sensitive, etc.)
// We also need to be able to filter some elements by some of their fields
// (e.g xp with operator "John Doe"). This will be
//
// When typing in search, all the lists are being filtered in real time for all the elements returned
// by the search (with a logical OR). Someone can then pin an element to the filter bar by selecting it
// in one of the lists, or by clicking on the search result dropdown.

/* Description of the project:
 *
 * We want to create a web app that allows users to browse the constructome. The constructome is a
 * database of synthetic biology parts, as well as the experiments that have been performed on them.
 *
 *   - Each experiment contains some info + a list of implemented recipes
 *   - Each recipe contains some info + a list of Plasmids
 *   - Each plasmid can be either an L2 or an L1, which themselves contain a list of TUs (transciption unit). 
 *     An L1 contains only one TU, an L2 contains multiple TUs
 *   - A TU contains a list of parts
 *
 * Each of these element type corresponds to a component that has a short "tag" version and an expanded version.
 * The tag version is a small component that can be displayed in a list, and the expanded version is a larger more detailed view that 
 * shows all the info about the component, (including the nested components, unexpanded by default).
 *
 * The app should allow users to browse the constructome in a hierarchical way, and to filter the displayed elements using a tag system.
 * It's based on list representation (organized as columns). All of the types of elements described above can be displayed in a list, and can be filtered.
 * For example, we have a list of experiments, and a list of recipes. 
 * By default, these 2 are the only lists displayed in their "list" form (but can also be folded easily). The other lists are just
 * displayed in their folded version: a one line version that shows their name and number of elements (e.g "Parts (57)", "L2s (12)", etc.).
 * When clicking on the folded list name, the list is expanded (a new column is dislayed).
 *
 * The user can also search for any of these elements, and filter any list to only show elements that contain them (including this element itself, as well).
 * The filtering relies on the filter component and the filter bar. We pin filters to the filter bar, and this affect which elements 
 * are displayed in the lists.
 * A filter is a pair of field and value, + some modifiers (exact match, fuzzy, case sensitive, inverse, etc.). A special case is the "name" or "id" field, which 
 * uniquely identifies an element, and can be used to filter the list to only show this element and its parents + children. It's just a special case in terms of how it's displayed: 
 * it's displayed as the tag version of the element.
 * Other filters (on fields) are displayed as "elmt_type.field = value" (e.g "xp.operator = John Doe").
 *
 * There are 2 way to add a filter: either by typing in the search bar, or by clicking on an element's field in one of the lists.
 * The search bar is a fuzzy finder that will suggest in real time, as we type, a list of filters that match the search. (Use fuse.js) 
 * We therefore need to first generate a list of all the possible filters (all the fields of all the elements), and let them be brought up by fuse.js.
 *
 * the second way to add a filter is by clicking on an element's field in one of the lists. This will add the corresponding filter to the filter bar.
 * For example, if we click on the "operator" field of an xp that has John Doe as it's operator, it will add the filter "xp.operator = John Doe" to the filter bar. 
 * (We can later add a modifier to this filter, such as "exact match" or "case sensitive".)
 *
 *
 * We have a REST server (localhost:4321) that serves the data. We can get all the data we need by doing a GET request to the following endpoints:
 *  - /xps
 *  - /recipes
 *  - /L2s
 *  - /L1s
 *  - /TUs
 *  - /parts
 *
 * Each of these endpoints returns a list of elements of the corresponding type. Each element has a unique name field.
 *
 *
 *

/*════════════════════════════════════════════════════════════════════════════════*/
