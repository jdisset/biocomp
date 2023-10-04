import React, { useState, useEffect, useCallback } from "react";

const ParamInput = ({
  isDragging,
  startMousePosition,
  startValue,
  objPosition,
  onChange,
  onClose,

  pname,
  tunableData,
  setPValue,
  pvalue,

  vmin = 0,
  vmax = 1,
}) => {
  const slideIncrements = [0.001, 0.01, 0.05, 0.1];
  const INCREMENT_REGION_HEIGHT = 20;

  const [value, setValue] = useState(startValue);
  const [slideIncId, setSlideIncId] = useState(0);

  const onEnd = () => {
    onClose();
  };

  const onUpdate = (event) => {
    if (isDragging) {
      setSlideIncId(
        Math.round(Math.abs(event.clientY - startMousePosition.y) / INCREMENT_REGION_HEIGHT),
      );
      const coef = slideIncrements[Math.min(slideIncId, slideIncrements.length - 1)];
      const newval = (event.clientX - startMousePosition.x) * coef + startValue;
      const v = Math.max(vmin, Math.min(vmax, newval));
      if (v !== v) return; // detect nan
      setValue(v);
      onChange(v);
    }
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", onUpdate);
      document.addEventListener("mouseup", onEnd);
    } else {
      //document.removeEventListener("mousemove", onUpdate);
      //document.removeEventListener("mouseup", onEnd);
    }
    return () => {
      document.removeEventListener("mousemove", onUpdate);
      document.removeEventListener("mouseup", onEnd);
    };
  }, [isDragging]);

  const W = 120;

  useEffect(() => {
    if (tunableData) {
      for (const [path, i, name, value] of tunableData) {
        if (name == pname) setPValue(value);
      }
    }
  }, [tunableData]);

  useEffect(() => {
    if (tunableData) {
      const new_tunable = tunableData.map(([path, i, name, value]) => {
        return [path, i, name, name == pname ? pvalue : value];
      });
      tunableData.updateMyParams(new_tunable);
    }
  }, [pvalue]);


  const incrementRegions = slideIncrements.map((inc, i) => {
    return (
      <div
        className="increment-region"
        key={i}
        style={{
          position: "absolute",
          left: 0,
          top: i * INCREMENT_REGION_HEIGHT,
          width: W,
          height: INCREMENT_REGION_HEIGHT,
          //backgroundColor: "rgba(255, 255, 255, 0.5)",
          fontSize: 5,
          verticalAlign: "middle",
          border: "0.1px solid rgba(0, 0, 0, 0.25)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: "10%",
            top: "50%",
            transform: "translateY(-50%)",
          }}
        >
          x{inc}
        </div>
      </div>
    );
  });

  return (
    <div
      style={{
        position: "absolute",
        display: isDragging ? "block" : "none",
        width: 5,
        height: 5,
        borderRadius: 15,
        left: objPosition.x - 9,
        top: objPosition.y - 9,
        padding: 0,
        margin: 0,
        verticalAlign: "middle",
        backgroundColor: "rgba(230, 8, 25, 0.75)",
        fontSize: 5,
      }}
    ></div>
  );
};

export default ParamInput;
