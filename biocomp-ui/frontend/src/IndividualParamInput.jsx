import React, { useState, useEffect, useCallback } from "react";

const IndividualParamInput = ({
  mouseDownEvent,
  clearMouseDownEvent,

  objpos,

  pname,
  subname,

  tunableData,
  updateParams,

  pvalue,
  setPValue,

  vmin = 0,
  vmax = 1,
}) => {
  const slideIncrements = [0.001, 0.01, 0.05, 0.1];
  const INCREMENT_REGION_HEIGHT = 20;
  const [slideIncId, setSlideIncId] = useState(0);
  const [startValue, setStartValue] = useState(0);

  const onEnd = () => {
    clearMouseDownEvent();
  };

  const onUpdate = (event) => {
    if (mouseDownEvent) {
      setSlideIncId(
        Math.round(Math.abs(event.clientY - mouseDownEvent.clientY) / INCREMENT_REGION_HEIGHT),
      );
      const coef = slideIncrements[Math.min(slideIncId, slideIncrements.length - 1)];
      const newval = (event.clientX - mouseDownEvent.clientX) * coef + startValue;
      const v = Math.max(vmin, Math.min(vmax, newval));
      if (v !== v) return; // detect nan
      setVal(v);
    }
  };

  useEffect(() => {
    if (mouseDownEvent) {
      document.addEventListener("mousemove", onUpdate);
      document.addEventListener("mouseup", onEnd);
      document.body.style.cursor = 'ew-resize';
      setStartValue(getVal());
    } else {
      document.body.style.cursor = 'auto';
    }
    return () => {
      document.body.style.cursor = 'auto';
      document.removeEventListener("mousemove", onUpdate);
      document.removeEventListener("mouseup", onEnd);
    };
  }, [mouseDownEvent]);


  useEffect(() => {
    if (tunableData) {
      for (const [path, i, name, value] of tunableData) {
        if (name == pname) {
          if (pvalue != value) setPValue(value);
        }
      }
    }
  }, [tunableData]);

  const getVal = () => {
    if (subname !== undefined && subname !== null) {
      return pvalue[subname];
    } else {
      return pvalue;
    }
  };

  const setVal = (v) => {
    let new_pvalue;
    if (subname !== undefined && subname !== null) {
      new_pvalue = [...pvalue];
      new_pvalue[subname] = v;
    }
    else {
      new_pvalue = v;
    }
    setPValue(new_pvalue);
  };


  useEffect(() => {
    if (tunableData) {
      const new_tunable = tunableData.map(([path, i, name, value]) => {
        return [path, i, name, name == pname ? pvalue: value];
      });
      updateParams(new_tunable);
    }
  }, [pvalue]);

  return (
    <div
      style={{
        position: "fixed",
        display: mouseDownEvent ? "block" : "none",
        width: 5,
        height: 5,
        borderRadius: 15,
        left: objpos.x,
        top: objpos.y,
        padding: 0,
        margin: 0,
        verticalAlign: "middle",
        backgroundColor: "rgba(230, 8, 25, 0.75)",
        fontSize: 5,
        cursor: mouseDownEvent ? "ew-resize" : "auto"
      }}
    ></div>

  );
};

export default IndividualParamInput;
