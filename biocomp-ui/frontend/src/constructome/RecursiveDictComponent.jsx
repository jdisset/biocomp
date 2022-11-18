import React, { useState, useEffect, useRef, useCallback } from "react";

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

export default RecursiveDict;
