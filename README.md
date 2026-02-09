# TV\_GNN

This experiment requires configuring the necessary environment according to requirements.txt, and then running the ./run\_scripts.sh script in the TV\_GNN directory to obtain the experimental results. Below is a brief introduction to the commands in this script.

1\. python lifted\_sampling\_fo2/weight\_TV.py :

This command will generate an Excel file in the ./lifted\_sampling\_fo2/data1 folder, showing the relationship between weight and TV-distance. We will use the weights corresponding to TV-distance=0.0,...,1.0 to prepare for subsequent sampling.

2\. python ./lifted\_sampling\_fo2/sampler.py -i ./lifted\_sampling\_fo2/models/color1.wfomcs -k 100000 -o ./lifted\_sampling\_fo2/outputs/color1/num100k\_mln/domain10 :

This command will sample using different weight pairs based on different TV-distances, with a sample size of 10k positive and 10k negative samples. Subsequent training and validation sets will be randomly sampled from this sample.When running the modify command, it is necessary to ensure that the sixth and seventh lines of the color1.wfomcs file:

V = 0

0 1 aux

3\. python ./lifted\_sampling\_fo2/dump.py :

This command converts the sampled pkl files into JSON files, facilitating data retrieval for subsequent GNN networks.

4\. python ./GNN/gan/train.py ./GNN/wfomi\_data/json/color1\_100k\_mln/domain10/tv0.0 --train-size 100 --val-size 10000 --gpu 0:

This command indicates that training will be performed on data with TV-distance=0. The training set and validation set contain 100 positive and 10000 negative samples, respectively. These samples are randomly selected from a previously sampled 10k dataset. The training results are ultimately stored in ./GNN/wfomi\_data/json/color1\_100\_mln/domain10

5\. python ./GNN/gan/plot\_ave\_acc.py :

This command generates a graph based on the average of five training iterations per TV-distance.

