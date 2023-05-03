```{eval-rst}
.. include:: sinebow.rst
```

<img src="../resources/skoots_train_api_flow.pdf" align="center" class="only-light"/>
<img src="../resources/skoots_train_api_flow_inverted.pdf" align="center" class="only-dark"/>

```{eval-rst} 
.. autofunction:: skoots.lib.embedding_to_prob.baked_embed_to_prob
   :noindex:
.. autofunction:: skoots.lib.skeleton.bake_skeleton
   :noindex:
.. autofunction:: skoots.lib.flood_fill.efficient_flood_fill
   :noindex:
.. autofunction:: skoots.lib.vector_to_embedding.vector_to_embedding
   :noindex:

.. autoclass:: skoots.train.loss.tversky
   :noindex:
    :special-members: __init__
    :members: forward
    :private-members:  _tversky
```