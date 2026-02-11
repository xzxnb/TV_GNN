#  TV\_GNN Experiment Guide

 This experiment requires configuring the necessary environment according to `requirements.txt`, and then running the `./run_scripts.sh` script in the TV_GNN directory to obtain the experimental results. Below, I will briefly describe the experimental steps and the meaning of the commands in the script.

## Generate a table of TV-distance and weight.
The command  `python lifted_sampling_fo2/weight_TV.py` will generate an Excel file in the `./lifted_sampling_fo2/data1` folder, showing the relationship between weight and TV-distance. We will use the weights corresponding to TV-distance=0.0,...,1.0 to prepare for subsequent sampling.

## Sample graphs
The command `python ./lifted_sampling_fo2/sampler.py -i ./lifted_sampling_fo2/models/color1.wfomcs -k 100000 -o ./lifted_sampling_fo2/outputs/color1/num100k_mln/domain10` will sample using different weight pairs based on different TV-distances, with a sample size of 10k positive and 10k negative samples. The weight information used for sampling is located in the `./lifted_sampling_fo2/outputs/color1/num100k_mln\domain10` folder, These weights were selected from the table generated earlier in `./lifted_sampling_fo2/data1`, based on the principle of being closest to the required TV-distance=0.0,...,1.0. Subsequent training and validation sets will be randomly sampled from this sample. When this command is run, the sixth and seventh lines of the `./lifted_sampling_fo2/models/color1.wfomcs file` needs to be initialized: 
* `V = 0`
* `0 1 aux`

## Convert to JSON file
The command `python ./lifted_sampling_fo2/dump.py` converts the sampled pkl files in the ./lifted_sampling_fo2/outputs/color1/num100k_mln/domain10` into JSON files, facilitating data retrieval for subsequent GNN networks. 

## Training GNN network
We can then use many similar commands to set different training and test set sizes to train the GNN, e.g. `python ./GNN/gan/train.py ./GNN/wfomi_data/json/color1_100k_mln/domain10/tv0.0 --train-size 100 --val-size 10000 --gpu 0`.  This command indicates that training will be performed on data with TV-distance=0. The training set contains 100 positive and negative samples, and validation set contains 10000 positive and negative samples. These samples are randomly selected from a previously sampled 10k dataset. The training results are ultimately stored in `./GNN/wfomi_data/json/color1_100_mln/domain10`

## Draw the experimental results
We conducted five experiments for each TV-distance using different training and test machine sizes. This command `python ./GNN/gan/plot_ave_acc.py` will plot the average highest test set accuracy versus TV-distance for these five experiments.

