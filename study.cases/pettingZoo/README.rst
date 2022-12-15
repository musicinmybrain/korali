Reinforcement Learning examples on pettingZoo
==============================================

This folders contain a ready-to-use setup to run pettingZoo. 

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
   
   % pettingZoo
   Waterworld


To run any of these, use the following example:

.. code-block:: bash

   python run-vracer.py --env Waterworld 

Producing a movie:
-------------------------

To generate a movie that displays the outcome of a particular trained policy, use the following commands:

.. code-block:: bash
   
   mkdir images
   python genMovie.py --env Waterworld --input _result_vracer_Waterworld_Clipped_Normal_0.0001_0.1_0.0
   cd images
   ffmpeg -framerate 180 -i image_%d.png output.mp4
   
The command will read the result of training an `Waterworld` environment from the `_result_vracer_Waterworld_Clipped_Normal_0.0001_0.1_0.0` folder and store images `.png` in the `images` folder and afterwards we animate the images with the last command.

