#!/bin/bash

exp=mt5_artificial_punct_fix

mkdir -p split/gold split/${exp} werr/${exp} merr/${exp} interwoven/${exp}
for f in gold/*; do
    sed -i '' '/^\\p/d' $f   # remove POS line in gold file
    f1=`basename $f`
    lang="$(cut -d'-' -f1 <<<$f1)"
    outputs="../test_preds/${exp}/${lang}-test-track2-covered.txt"
    f2=`basename $outputs`
    errors="${exp}-${lang}.txt"

    # uncomment to count errors by word
    rm -f -- werr/${exp}/${errors} 
    ./counterrors.pl $f $outputs > werr/${exp}/${errors}     
    
    # uncomment to count errors by morpheme
    sed 's/-/ /g' < $f > split/gold/${f1}
    sed 's/-/ /g' < $outputs > split/${exp}/${f2}
    rm -f -- merr/${exp}/${errors} 
    ./counterrors.pl split/gold/${f1} split/${exp}/${f2} > merr/${exp}/${errors}
    
    # uncomment next two lines to interleave gold and pred lines
    rm -f -- interwoven/${exp}/${errors}                     
    python interleave_preds.py $f $outputs > interwoven/${exp}/${errors}
done
