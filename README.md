# The Convolutional Tsetlin Machine (coming soon)
The Convolutional Tsetlin Machine learns interpretable filters using propositional formulae. It is an interpretable alternative to convolutional neural networks.

## Demo

## Learning Behaviour
The below figure depicts training and test accuracy for the Convolutional Tsetlin Machine on MNIST, epoch-by-epoch, in a single run. 
![Figure 4](https://github.com/olegranmo/blob/blob/master/performance_by_epoch_MNIST.png)
Test accuracy peaks at 99.50% after 168 epochs and 99.51% after 327 epochs. Further, test accuracy climbs quickly in the first epochs, passing 99% already in epoch 2. Training accuracy approaches 100%. 

## Step-by-Step Walkthrough of Inference

### Recognition
Rather than providing hand-crafted features which can be used for image classification, the CTM learns feature detectors. We will explain the workings of the CTM by an illustrative example of noisy 2D XOR recognition and learning. Consider the CTM depicted in the below figure. 
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/Recognition.png">
</p>
It consists of four positive clauses which represent XOR patterns that must be present in a positive example image (positive features) and four negative clauses which represent patterns that will not trigger a positive image classification (negative features). The number of positive and negative clauses is a user-defined parameter. The bit patterns inside each clause are represented by the output of four Tsetlin Automata, one for each bit in a 2x2 filter.

### Learning

Consider the 3x3 image shown below. The filter represented by the second positive clause matches the patch in the top-right corner of the image and it is the only clause with output 1; similarly, none of the negative clauses respond since their patterns do not match the pattern found in the current patch 
<p align="center">
  <img width="105%" src="https://github.com/olegranmo/blob/blob/master/Learning.png">
</p>
Thus, the Tsetlin Machine’s combined output is 1. Learning of feature detectors proceeds as follows: With the CTM’s threshold value set to T = 2, the probability of feedback is (T-v)/(2T)=0.25, and thus learning taking place, which pushes the CTM’s output v towards T=2. Note that Type I feedback reinforces true positive output and reduces false negative output whereas Type II feedback reduces false positive output.

### Goal State
A subsequent state of the CTM is shown below.
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/Goal_State.png">
</p>
Note that there are now two positive clauses which detect their pattern in the top-right corner patch. The combined output of all clauses is 2; thus, no further learning is necessary for the detection of the XOR pattern in this patch. Also, the location of the occurrence of each pattern is included. The location information uses a bit representation as follows: Suppose an XOR pattern occurs at the three X-coordinates 1, 4, and 6. For the corresponding binary location representation, these coordinates are considered thresholds: If a coordinate is greater than a threshold, then the corresponding bit in the binary representation will be 0; otherwise, it is set to 1. Thus, the representation of the X-coordinates 1, 4, and 6 will be ‘111’, ‘011’ and ‘001’, respectively. These representations of the location of 2x2 patterns are also learned by TAs.

## Licence

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
