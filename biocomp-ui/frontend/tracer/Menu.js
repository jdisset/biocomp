import React, { useEffect, useRef, useState, useMemo } from "react";
import ReactDOM from "react-dom";
import { layoutData, pointData } from "./data"; // import your data
import { COLORS } from "./constants";
import "./style.css";
import { FiMenu } from "react-icons/fi"; // import an icon for the button

const Menu = ({ settings, setSettings }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(true);

  return (
    <div className="mainmenu">
      <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="menu-button">
        <FiMenu size={24} />
      </button>
      <div className={`mainmenu ${isMenuOpen ? "" : "collapsed"}`}>
        <div className="nav-tools">
        </div>
        <div className="color-mode">
          <label>Color Mode</label>
          <select
            value={settings.colorMode}
            onChange={(e) => setSettings({ ...settings, colorMode: e.target.value })}
          >
            <option value="solid">Solid</option>
            <option value="gradient">Gradient</option>
            <option value="selectedAxis">Selected Axis</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default Menu;
