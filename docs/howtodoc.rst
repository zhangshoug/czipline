Maintainance of the documentation
=================================

The Documentation is generated with Sphinx from ReStructuredTxt files ('.rst'). 
Many sections however also exist and are maintained as notebooks, using basic formatting.
The conversion from notebook to rst is done via a script (:download:`take a look <scripts/nbconvert_to_rst.py>`)
and has been included as a Makefile command (make rst).

The workflow is as follow:

.. code-block:: bash

    1. cd docs 
    2. ... # edit notebooks in notebooks/
    3. ... # edit rst files 
    4. make rst  # convert every notebook in docs/notebooks to rst in docs/_notebooks_rst
    5. make html  # this could also be combine above in make rst html
    6. ... # check the result in docs/_build/html/index.html
    7. ... # iterate until you are happy with the result
    8. git add / rm / ci  # commit the change
    9. git push  # push to github

Pushing to github will update the doc at readthedocs automatically.

.. note:: Step 4 will work only on unix system because bash is involved in one of the scripts (this could actually be written in python easily)
          There might also be other dependencies involved, maybe even the ipython version (did that with the latest 3.0.0).

.. note:: readthedocs will re-compile the rst files to html, 
          so that steps 5-6 using your local sphinx installation are only 
          for you to check the results before pushing.

.. note:: To compile locally with sphinx, you need to download 
          sphinx of course, but also numpydoc (which parse numpy-like docstrings)
          e.g. "pip -r docs/readthedocs-pip-requirements.txt"
