# Clotho data loader

Welcome to Clotho data loader repository. This repository has the necessary code for
using the DataLoader class from PyTorch package (`torch.utils.data.dataloader.DataLoader`)
with the Clotho dataset. 

You can use the present data loader of Clotho directly with the examples created by the
[Clotho baseline dataset repository](https://github.com/dr-costas/clotho-baseline-dataset). 

If you are looking at this README file, then I suppose that you already know what is a
DataLoader from PyTorch. Nevertheless, the Clotho dataset has sequences as inputs and outputs,
and each sequence is of arbitrary length (15 to 30 seconds for the input and 8 to 20 words 
for the output). For that reason, this data loader already provides a collate function. 

## Collate function

To be able to use the sequences of Clotho in a batch, you most likely will need some kind of padding
policy. This repository already offers a collate function to be used with the Clotho data. 

With the provided collate function, you can choose to either: 

* pad the data with zeros (for input audio data) and end-of-sequence symbol (for the output/words), 
to the length of the longest input (for the inputs) and output (for the outputs) sequence in
tha batch
* truncate the input and the output to the minimum length of the input and output in the batch, and
* use a constant length for input and output, and either truncate or pad. 

Enjoy and if you have any issues, please let me know in the issue section. 