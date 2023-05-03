.. include:: sinebow.rst

:nge-light-blue:`Data Transforms API Flow`
==========================================

This gives the rough flow of data outlining the data augmetnation procedure to train a SKOOTS model.

.. image:: ../resources/skoots_transforms_api_flow.pdf
   :width: 800
   :class: only-light

.. image:: ../resources/skoots_transforms_api_flow_inverted.pdf
   :width: 800
   :class: only-dark

.. autoclass:: skoots.train.dataloader.dataset
   :members:
   :private-members:
   :noindex:

.. autoclass:: skoots.train.dataloader.MultiDataset
   :members:
   :private-members:
   :undoc-members:
   :noindex:

.. autofunction:: skoots.lib.skeleton.skeleton_to_mask
   :noindex:
.. autofunction:: skoots.lib.skeleton.bake_skeleton
   :noindex:
