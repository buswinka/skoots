.. include:: sinebow.rst

:nge-light-blue:`Training API Flow`
==============================

Presented here is the rough flow of data from necessary to train a SKOOTS segmentation model. This is markedly different
from evaluation, which is presented elsewhere.

.. image:: ../../resources/skoots_train_api_flow.pdf
   :width: 800
   :class: only-light

.. image:: ../../resources/skoots_train_api_flow_inverted.pdf
   :width: 800
   :class: only-dark

.. autofunction:: skoots.lib.embedding_to_prob.baked_embed_to_prob
   :noindex:
.. autofunction:: skoots.lib.skeleton.bake_skeleton
   :noindex:
.. autofunction:: skoots.lib.flood_fill.efficient_flood_fill
   :noindex:
.. autofunction:: skoots.lib.vector_to_embedding.vector_to_embedding
   :noindex:

.. autoclass:: skoots.train.loss.tversky
    :special-members: __init__
    :members: forward
    :private-members:  _tversky
    :noindex: