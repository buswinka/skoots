.. include:: sinebow.rst

:nge-light-blue:`Evalutation API Flow`
======================================

This gives the rough flow of data from one function to the next, neccessary to
perform instance segmentation with SKOOTS. Each image is expected to be a 5D tensor with
shape (B, C, X, Y, Z).

.. image:: ../resources/skoots_eval_api_flow.pdf
   :width: 800
   :class: only-light

.. image:: ../resources/skoots_eval_api_flow_inverted.pdf
   :width: 800
   :class: only-dark

.. autofunction:: skoots.lib.flood_fill.efficient_flood_fill
   :noindex:
.. autofunction:: skoots.lib.vector_to_embedding.vector_to_embedding
   :noindex:
.. autofunction:: skoots.lib.skeleton.index_skeleton_by_embed
   :noindex:
