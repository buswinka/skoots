.. include:: sinebow.rst

:nge-red:`SKOOTS`
=================

:nge-red:`Sk` eleton
:nge-yellow:`O` riented
:nge-mint-green:`O` bjec
:nge-green:`t`
:nge-green:`S` egmentation

.. image:: ../../resources/skooting_in_progress_v2.png

SKOOTS is an approach at high resolution, 3D mitochondria segmentation. Bridging the gap between boundary-prediction networks
and flow based approaches, SKOOTS offers robust segmentation of high resolution mitochondria in previously difficult situations.

SKOOTS is presented as a python library with documentation provided here. Created with a functional interface, see the
API Flow guide for instructions on how to incorporate a SKOOTS into a segmentation pipeline, and our example scripts for
implementation.

.. grid:: 1 1 2 2
    :gutter: 1

    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 2

            .. grid-item-card::

                .. toctree::
                    :caption: Basics
                    :maxdepth: 3

                    basics.rst

            .. grid-item-card::

                .. toctree::
                    :caption: Evaluation
                    :maxdepth: 3

                    evaluation



    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 2

            .. grid-item-card::

                .. toctree::
                    :caption: API Flow
                    :maxdepth: 2

                    eval_api_flow.rst
                    train_api_flow.rst
                    transforms_api_flow

            .. grid-item-card::

                .. toctree::
                    :caption: API Reference
                    :maxdepth: 2

                    api_reference

            .. grid-item-card::

                .. toctree::
                    :caption: Training
                    :maxdepth: 3

                    simple_train
                    detailed_training
                    custom_train_engine
