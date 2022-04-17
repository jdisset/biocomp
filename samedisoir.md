"Now let's generate all the possible sequestron graphs from our library".
I wrote that.

I am not sure that's the right aproach, but let's explore it fully.
We have:
- gdf: the l1 graph. 
- seq: the possible sequestrons for the l0s we have, each row linking 2 specific parts together.

from this we can deduce, for a given gdf, the actual compute graph.
We start from the output.

