Reinforcement Learning examples on MAgent2
==============================================

This folders contain a ready-to-use setup to run `MAgent2 <https://github.com/Farama-Foundation/MAgent2>`_. 

Pre-Requisites:
------------------
None.

Installation:
------------------
./install_deps.sh

Potential installation errors :
---------------------------------
.. code-block::bash
   error: command 'swig' failed with exit status 1
If this error appears, please follow step 1 and 2 to install the latest version of swig on: http://swig.org/svn.html 

Running an environment:
-------------------------

Any of the following environments are available for testing:

.. code-block:: bash
   
   % magent2
   Battle


To run any of these, use the following example:

.. code-block:: bash

   python run-dvracer.py --env Battle 


