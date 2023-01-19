/*────────────────────────────────▼     DESCRIPTION     ▼────────────────────────────────*/
/*
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

/*════════════════════════════════════════════════════════════════════════════════*/

/*───────────────────────────────▼     import     ▼───────────────────────────────*/

import React, { useState } from "react";
import axios from "axios";
import Fuse from "fuse.js";

/*════════════════════════════════════════════════════════════════════════════════*/

const Filter = ({ filter, onFilterRemove, onFilterAdd }) => (
  <div className="filter">
    {filter.field}: {filter.value}
    {onFilterRemove && (
      <button onClick={() => onFilterRemove(filter)}>X</button>
    )}
    {onFilterAdd && (
      <button onClick={() => onFilterAdd(filter)}>+</button>
    )}
  </div>
);

const FilterBar = ({ filters, onFilterRemove }) => (
  <div className="filter-bar">
    {filters.map(filter => (
      <Filter
        key={filter.field}
        filter={filter}
        onFilterRemove={onFilterRemove}
      />
    ))}
  </div>
);

const SearchBar = ({ query, onQueryChange, onFilterAdd }) => (
  <div className="search-bar">
    <input
      type="text"
      placeholder="Search for a filter"
      value={query}
      onChange={onQueryChange}
    />
    <SuggestedFilters
      query={query}
      onFilterAdd={onFilterAdd}
    />
  </div>
);

const SuggestedFilters = ({ query, onFilterAdd, allFilters }) => (
  <div className="suggested-filters">
    {/* Use Fuse to search for filters that match the search query */}
    const fuse = new Fuse(allFilters, {
      keys: ['field', 'value'],
    });
    const results = fuse.search(query);
    {results.map(filter => (
      <Filter
        key={filter.field}
        filter={filter}
        onFilterAdd={onFilterAdd}
      />
    ))}
  </div>
);

const Element = ({ element, onFilterAdd }) => (
  <div className="element">
    {Object.keys(element).map(key => (
      <div key={key} onClick={() => onFilterAdd({ field: key, value: element[key] })}>
        <strong>{key}</strong>: {element[key]}
      </div>
    ))}
  </div>
);

const List = ({ name, elements, onFilterAdd }) => (
  <div className="list">
    <h3>{name}</h3>
    {elements.map(element => (
      <Element
        key={element.name}
        element={element}
        onFilterAdd={onFilterAdd}
      />
    ))}
  </div>
);

const App = () => {
  // Initialize the state for the different lists of data
  const [xps, setXps] = useState([]);
  const [recipes, setRecipes] = useState([]);
  const [l2s, setL2s] = useState([]);
  const [l1s, setL1s] = useState([]);
  const [tus, setTus] = useState([]);
  const [parts, setParts] = useState([]);

  // Initialize the state for the filters and search query
  const [filters, setFilters] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch the data from the REST server when the component is first mounted
  useEffect(() => {
    axios.get("localhost:4321/xps").then((res) => setXps(res.data));
    axios.get("localhost:4321/recipes").then((res) => setRecipes(res.data));
    axios.get("localhost:4321/l2s").then((res) => setL2s(res.data));
    axios.get("localhost:4321/l1s").then((res) => setL1s(res.data));
    axios.get("localhost:4321/tus").then((res) => setTus(res.data));
    axios.get("localhost:4321/parts").then((res) => setParts(res.data));
  }, []);

  // Handle changes to the search query by updating the state
  const handleSearchQueryChange = (event) => {
    setSearchQuery(event.target.value);

    // Use Fuse to search for filters that match the search query
    const fuse = new Fuse(filters, {
      keys: ["field", "value"],
    });
    const results = fuse.search(searchQuery);

    // Update the list of suggested filters
    setSuggestedFilters(results);
  };

  // Handle adding a filter to the filter bar
  const handleFilterAdd = (filter) => {
    setFilters([...filters, filter]);
  };

  // Handle removing a filter from the filter bar
  const handleFilterRemove = (filter) => {
    setFilters(filters.filter((f) => f !== filter));
  };

  return (
    <div className="app">
      <SearchBar
        query={searchQuery}
        onQueryChange={handleSearchQueryChange}
        onFilterAdd={handleFilterAdd}
      />
      <FilterBar filters={filters} onFilterRemove={handleFilterRemove} />
      <List name="Experiments" elements={xps} onFilterAdd={handleFilterAdd} />
      <List name="Recipes" elements={recipes} onFilterAdd={handleFilterAdd} />
      <List name="L2s" elements={l2s} onFilterAdd={handleFilterAdd} />
      <List name="L1s" elements={l1s} onFilterAdd={handleFilterAdd} />
      <List name="TUs" elements={tus} onFilterAdd={handleFilterAdd} />
      <List name="Parts" elements={parts} onFilterAdd={handleFilterAdd} />
    </div>
  );
};

export default App;
