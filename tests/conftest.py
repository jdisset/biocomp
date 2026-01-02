"""pytest configuration for biocomp tests."""
import os

# Disable JAX compilation cache BEFORE importing JAX
# This is necessary because JAX's cache can cause incorrect behavior with NSGA2
# due to how closures over objects interact with JIT compilation caching.
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "0"
