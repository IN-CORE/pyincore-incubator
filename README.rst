pyincore-incubator
========================

**pyIncore-incubator** is a component of IN-CORE that allows users to extend
the core set of analyses provided by pyincore. New analyses in
pyincore-incubator extend the pyincore BaseAnalysis class and can be chained
with existing pyincore analyses to add new functionality. The incubator gives
the community around the IN-CORE project a forum and set of resources for
innovation and investigation of new ideas and alternative ideas.

Installation with conda
-----------------------

Installing **pyincore-incubator** with Conda is officially supported by IN-CORE development team.

To add `conda-forge <https://conda-forge.org/>`__  channel to your environment, run

.. code-block:: console

   conda config –-add channels conda-forge

To install **pyincore-incubator** package, run

.. code-block:: console

   conda install -c in-core pyincore-incubator


To update **pyIncore-incubator**, run

.. code-block:: console

   conda update -c in-core pyincore-incubator

You can find detail information at the
`Installation <https://incore.ncsa.illinois.edu/doc/incore/pyincore/install_pyincore.html>`__
section at IN-CORE manual.

Installation with pip
-----------------------

Installing **pyincore-incubator** with pip is **NOT supported** by IN-CORE development team.
Please use pip for installing pyincore-incubator at your discretion.

**Installing pyincore-incubator with pip is only tested on the linux environment.**

**Prerequisite**

* GDAL C library must be installed to install pyincore-incubator. (for Ubuntu, **gdal-bin** and **libgdal-dev**)

To install **pyincore-incubator** package, run

.. code-block:: console

   pip install pyincore-incubator


Testing and Running
-------------------

Please read the `Testing and
Running <https://incore.ncsa.illinois.edu/doc/incore/pyincore/running.html>`__
section at IN-CORE manual.

Documentation
-------------

For reference, the documentation of pyincore can be found below since
pyincore-incubator adds new analyses to the core set provided by pyincore.

**pyIncore** documentation can be found at
https://incore.ncsa.illinois.edu/doc/incore/pyincore.html

**pyIncore** technical reference (API) can be found at
https://incore.ncsa.illinois.edu/doc/pyincore/.

Acknowledgement
---------------

This work herein was supported by the National Institute of Standards
and Technology (NIST) (Award No. 70NANB15H044). This support is
gratefully acknowledged. The views expressed in this work are those of
the authors and do not necessarily reflect the views of NIST.
