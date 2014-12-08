# package documentation to upload to python hosted
# to be executed from docs/

MIRROR=1
HTML=_build/html
ZIP=pythonhosteddoc.zip

# remove exsiting doc 
rm $ZIP -f

if [ $MIRROR == 1 ] ; then

    # Just create a mirror
    mkdir tmp # temporary directory
    cp mirror.html tmp/index.html
    cd tmp
    zip $ZIP index.html
    cd ..
    mv tmp/$ZIP .
    rm tmp -rf # remove tmp dir

else

    # Or upload the full doc
    cd $HTML
    zip $ZIP * -r
    cd -
    mv $HTML/$ZIP .

fi

