.. include:: sinebow.rst

.. raw:: html

        <h1>
            <span style="color: #e72333">SK</span><span style="color: #e7a316">O</span><span style="color: #73bf85">OT</span><span style="color: #5a9432">S</span>
        </h1>
        <h4>
            <span style="color: #e72333">Sk</span>eleton <span style="color: #e7a316">O</span>riented <span style="color: #73bf85">O</span>bjec<span style="color: #73bf85">t</span> <span style="color: #5a9432">S</span>egmentation
        </h4>


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
                    :maxdepth: 1

                    quickstart
                    installation
                    quickstart_train
                    quickstart_evaluation
                    quickstart_inference




            .. grid-item-card::

                .. toctree::
                    :caption: Tutorials
                    :maxdepth: 3

                    simple_train
                    simple_inference
                    detailed_training
                    detailed_inference


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



