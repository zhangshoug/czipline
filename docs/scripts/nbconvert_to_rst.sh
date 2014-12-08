#!/bin/bash
# convert all ipython notebook to rst format

NOTEBOOKDIR=$1
RSTDIR=$2

#NOTEBOOKDIR=notebooks
#RSTDIR=_rst
indexfile=$RSTDIR/index.rst
TOC="" # --toc to include TOC
DOWN="" # --download to include download link
DOWN="--download" # --download to include download link

# start the index
echo "Documentation" > $indexfile
echo "-------------" >> $indexfile
echo "             " >> $indexfile
echo ".. toctree::"   >> $indexfile
echo "   :maxdepth: 3" >> $indexfile
echo "             " >> $indexfile


for file in $NOTEBOOKDIR/*ipynb ; do
    # skip files starting with _
    firstletter=`basename $file | cut -c1`
    if [ $firstletter == '_' ] ; then
	continue
    fi

    rstfile=$RSTDIR/`basename ${file%.ipynb}.rst`
    python scripts/nbconvert_to_rst.py $file $rstfile $TOC $DOWN

    # generate the index
    echo "   `basename $rstfile`" >> $indexfile
done
 

echo "index written to $indexfile" 
