import os
import streamlit.components.v1 as components

_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component("ned_component",
                                url="http://localhost:3001")
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("ned_component", path=build_dir)

def ned_component(name, key=None):
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    component_value = _component_func(name=name, key=key, default=0)
    return component_value

if not _RELEASE:
    import streamlit as st

    st.subheader("Component with constant args")
    num_clicks = ned_component("World")
    st.markdown("You've clicked %s times!" % int(num_clicks))
