# The Convolutional Tsetlin Machine (under construction)
The Convolutional Tsetlin Machine learns interpretable filters using propositional formulae. The formulae are composed by a collective of Tsetlin Automata. The Convolutional Tsetlin Machine is an interpretable alternative to convolutional neural networks.

## Learning Behaviour
The below figure depicts training and test accuracy for the Convolutional Tsetlin Machine on MNIST, epoch-by-epoch, in a single run. 
![Figure 4](https://github.com/olegranmo/blob/blob/master/performance_by_epoch_MNIST.png)
Test accuracy peaks at 99.50% after 168 epochs and 99.51% after 327 epochs. Further, it climbs quickly in the first epochs, passing 99% already in epoch 2. Training accuracy approaches 100%. 

## Step-by-Step Walkthrough of Inference

### Example Problem: 2D Noise XOR

I will use the 2D Noisy XOR dataset to demonstrate how the Convolutional Tsetlin Machine recognizes patterns and how these patterns are learnt from example images. The 2D Noisy XOR dataset contains 4x4 binary images, exemplified below.
<p align="center">
  <img width="12%" src="https://github.com/olegranmo/blob/blob/master/Example_Image.png">
</p>
The 9 bits of each image have been set randomly, expect for the four bits in the 2x2 patch found in the upper right corner of the image, marked by green bit values. These four bits reveal the class of the image:
<p align="center">
  <img width="25%" src="https://github.com/olegranmo/blob/blob/master/Patterns.png">
</p>
As seen, a diagonal line is associated with class 1, while a horizontal line is associated with class 0. The dataset thus captures a 2D version of the XOR-relation.

For a 4x4 image, there are four distinct 2x2 patches. Each patch consists of four propositional variables <img src="http://latex.codecogs.com/svg.latex?\mathbf{X} = [x_{1,1}, x_{2,1}, x_{1,2}, x_{2,2}]" border="0"/>, where each variable either takes the value 0 or 1: 
<p align="center">
  <img width="10%" src="https://github.com/olegranmo/blob/blob/master/Filter.png">
</p>

Rather than providing hand-crafted features which can be used for image classification, the Convolutional Tsetlin Machine learns feature detectors. For 2D Noisy XOR, the task is to learn a 2x2 convolution filter.

### The Tsetlin Automaton

The Convolutional Tsetlin Machine is based on the Tsetlin
Automaton, introduced by M. L. Tsetlin in 1961. The Tsetlin Automaton is one of the pioneering solutions to the well-known multi-armed bandit problem and the
first Finite State Learning Automaton. In all brevity, a Tsetlin Automaton performs actions sequentially in an environment. For each action, the environment responds stochastically with a penalty or a reward according to an unknown reward probability distribution. Based on the feedback from the environment, the goal of the Tsetlin Automaton is to, as quickly as possible, infer which action has the highest probability of eliciting a reward.

A Tsetlin Automaton is a finite state machine, and below you see a Tsetlin Automaton with 6 states, 3 states per action. 
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_1.png">
</p>
When the automaton is in states 1-3 (left side) it performs Action 1, and when it is in states 4-6 (right side) it performs Action 2.

<p>
The Tsetlin Automaton learns by changing state. As seen in the figure, each state transition is decided by whether the Tsetlin Automaton receives a penalty or reward. Being in state 3 in the figure (marked with a solid black circle), the Tsetlin Automaton would select Action 1. Assume this triggers a penalty from the environment. The Tsetlin Automaton would then move from state 3 to state 4:
</p>
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_2.png">
</p>
It is now on the right side of the state space and will be performing Action 2. This time, the Tsetlin Automaton receives a reward, updating its state accordingly:
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_3.png">
</p>
<p>
At this point it is quite confident that Action 2 is better than Action 1. Now, two or more consequtive penalties are needed to make the Tsetlin Automaton change its mind and switch back to performing Action 1 again.
</p>

<p>
This simple learning mechanism has some remarkable properties. It makes the Tsetlin Automaton act predictably, only changing action when switching between states 3 and 4. This supports stable collectives of many cooperating Tsetlin Automata, taking part in solving more complex problems. Further, the Tsetlin Automaton never stops learning, adapting to changes in the environment. This helps avoiding getting stuck at local optima.  Finally, the accuracy and speed of learning is controlled by the number of states. As the number of states and learning iterations approach infinity, the Tsetlin Automaton performs the optimal action with probability arbitrary close to unity. In other words, Tsetlin Automata learning is asymptotically optimal.
</p>


### Clause Formation

The Convolutional Tsetlin Machine uses easy-to-interpret conjunctive clauses as filters. The propositional variables <img src="http://latex.codecogs.com/svg.latex?x_{1,1}, x_{2,1}, x_{1,2}, x_{2,2}" border="0"/> and their negations <img src="http://latex.codecogs.com/svg.latex?\lnot{x_{1,1}}, \lnot{x_{2,1}}, \lnot{x_{1,2}}, \lnot{x_{2,2}}" border="0"/> are referred to as literals. A conjunctive clause is built by ANDing a selection of the literals.

<p align="center">
  <img width="90%" src="https://github.com/olegranmo/blob/blob/master/Example_Configuration_1a.png">
</p>

### Recognition
Rather than providing hand-crafted features which can be used for image classification, the Convolutional Tsetlin Machine (CTM) learns feature detectors. We will explain the workings of the CTM by an illustrative example of noisy 2D XOR recognition and learning. Consider the CTM depicted in the below figure. 
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

## Demo


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
