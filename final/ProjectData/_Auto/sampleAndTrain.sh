WIDTH=75
HEIGHT=100
NUMPOS=30
NUMNEG=151


opencv_createsamples -info positive/info.dat -vec out.vec -bg bg.txt -w $WIDTH -h $HEIGHT -show

opencv_traincascade -data . -vec out.vec -bg bg.txt -numPos $NUMPOS -numNeg $NUMNEG -numStages 3 -w $WIDTH -h $HEIGHT
