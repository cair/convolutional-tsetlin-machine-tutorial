# The Convolutional Tsetlin Machine (under construction)
The Convolutional Tsetlin Machine learns interpretable filters using propositional formulae. The propositional formulae are composed by a collective of Tsetlin Automata coordinated through a game. The Convolutional Tsetlin Machine is an interpretable alternative to convolutional neural networks.

## Learning Behaviour
The below figure depicts training and test accuracy for the Convolutional Tsetlin Machine on MNIST, epoch-by-epoch, in a single run. 
![Figure 4](https://github.com/olegranmo/blob/blob/master/performance_by_epoch_MNIST.png)
Test accuracy peaks at 99.50% after 168 epochs and 99.51% after 327 epochs. Further, it climbs quickly in the first epochs, passing 99% already in epoch 2. Training accuracy approaches 100%. 

## Step-by-Step Walkthrough of Inference

### Example Problem: 2D Noisy XOR

<p>
I will use the 2D Noisy XOR dataset to demonstrate how the Convolutional Tsetlin Machine recognizes patterns and how these patterns are learnt from example images. The 2D Noisy XOR dataset I consider contains 3x3 binary images. Below you see an example image.
</p>
<p align="center">
  <img width="12%" src="https://github.com/olegranmo/blob/blob/master/Example_Image.png">
</p>

<p>
The 9 bits of the image are assigned coordinates (x, y), so that the upper left bit is at position (1, 1), the bit to its right is at position (2, 1), while the bit below it is at position (1, 2), and so on. The 9 bit values are randomly set for each image, except for the four bits of the 2x2 patch in the upper right corner (marked by green bit values). These four bit values reveal the class of the image:
</p>
<p align="center">
  <img width="25%" src="https://github.com/olegranmo/blob/blob/master/Patterns.png">
</p>
<p>
A horizontal line is associated with class 0, while a diagonal line is associated with class 1. The dataset thus captures a 2D version of the XOR-relation.
</p>
<p>
For this example, I consider convolution with 2x2 filters. A convolutional learning mechanism employing 2x2 filters must learn the four patterns above as well as which class they belong to. Note that due to the XOR-relation, linear classifiers will face difficulties handling this task.
</p>
<p>
A 3x3 image contains four distinct 2x2 patches, located at different (x, y) coordinates within the image. One is located in the upper left of the image, at position (1, 1), another at position (2, 1), a third at position (1, 2), and the fourth at position (2, 2). The content of each patch is modelled with four propositional variables <img src="http://latex.codecogs.com/svg.latex?\mathbf{X} = [x_{1,1}, x_{2,1}, x_{1,2}, x_{2,2}]" border="0"/>. The coordinates of the variables (lower index) are the relative positions of the variables within the patch: 
<p align="center">
  <img width="10%" src="https://github.com/olegranmo/blob/blob/master/Filter.png">
</p>
<p>
This means that which image bit a propositional variable refers to depends both on the coordinates of the variable within the patch and the position of the patch within the image. As an example, the variable <img src="http://latex.codecogs.com/svg.latex?x_{1,1}" border="0"/> of the upper right patch refers to the bit at position (2, 1) in the 3x3 image.
</p>

<p>
For the 2D XOR dataset, the patch in the upper right part of the image, at position (2, 1), can be used to determine the class of the image, since it contains the discriminating 2D XOR-pattern. 
</p>

<p>
I will now explain how the Convolutional Tsetlin Machine solves the above pattern recognition task, going through the recognition and learning steps in detail.
</p>

### The Tsetlin Automaton

<p>
The Convolutional Tsetlin Machine is based on the Tsetlin Automaton, introduced by M. L. Tsetlin in 1961. The Tsetlin Automaton is one of the pioneering solutions to the well-known multi-armed bandit problem and the first Finite State Learning Automaton.
</p>

#### Two-Action Tsetlin Automata
<p>
A two-action Tsetlin Automaton chooses among two actions, Action 1 or Action 2, and performs these sequentially in an environment. For each action performed, the environment responds stochastically with a penalty or a reward, according to an unknown reward probability distribution <img src="http://latex.codecogs.com/svg.latex?R=\[r_1, r_2\]" border="0"/>. When Action 1 is performed, the environment responds with a reward with probability <img src="http://latex.codecogs.com/svg.latex?r_1" border="0"/>, otherwise, it responds with a penalty. For Action 2, the probability of a reward is <img src="http://latex.codecogs.com/svg.latex?r_2" border="0"/>.  By interacting with the environment, the goal of the Tsetlin Automaton is to, as quickly as possible, single in on the action that has the highest probability of eliciting a reward.
</p>

<p>
A Tsetlin Automaton is a finite state machine. Below you see a two-action Tsetlin Automaton with 6 states, 3 states per action.
</p>
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_1.png">
</p>
<p>
When the automaton is in states 1-3 (left half) it performs Action 1, and when it is in states 4-6 (right half) it performs Action 2.
</p>

<p>
The Tsetlin Automaton learns by changing state. Each state transition is decided by the feedback from the environment (reward or penalty). As shown in the figure above, a reward makes the Tsetlin Automaton change state away from the centre, while a penalty makes it change state towards the centre.
</p>

#### Example Run
<p>
The depicted Tsetlin Automaton is in state 3 (marked with a solid black circle). Accordingly, it selects Action 1. Assume this triggers a penalty from the environment. The Tsetlin Automaton then moves from state 3 to state 4:
</p>

<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_2.png">
</p>
It is now in the right half of states and therefore selects Action 2. This time, the Tsetlin Automaton receives a reward, updating its state accordingly:
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_3.png">
</p>
<p>
At this point, it is quite confident that Action 2 is better than Action 1. Indeed, at least two consequtive penalties are needed to make the Tsetlin Automaton change its mind and switch back to performing Action 1 again.
</p>

<p>
The above simple learning mechanism has some remarkable properties. It makes the Tsetlin Automaton act predictably, only changing action when switching between the two centre states. This supports stable collectives of many cooperating Tsetlin Automata, taking part in solving more complex problems. Further, the Tsetlin Automaton never stops learning. Therefore, it can adapt to changes in the environment. This helps avoiding getting stuck in local optima.  Finally, the accuracy and speed of learning is controlled by the number of states. By increasing the number of states and learning cycles towards infinity, the Tsetlin Automaton performs the optimal action with probability arbitrary close to unity. In other words, Tsetlin Automata learning is asymptotically optimal.
</p>


### Clause Formation

#### The Clause
<p>
The Convolutional Tsetlin Machine uses conjunctive clauses as filters. A clause is built by ANDing a selection of the available propositional variables <img src="http://latex.codecogs.com/svg.latex?x_{1,1}, x_{2,1}, x_{1,2}, x_{2,2}" border="0"/> and their negations <img src="http://latex.codecogs.com/svg.latex?\lnot{x_{1,1}}, \lnot{x_{2,1}}, \lnot{x_{1,2}}, \lnot{x_{2,2}}" border="0"/> (the propositional variables and their negations are referred to as literals). The clause <img src="http://latex.codecogs.com/svg.latex?C = {x_{1,1}} \land {x_{2,2}} \land  \lnot{x_{1,2}} \land \lnot{x_{1,2}}" border="0"/>, for instance, evaluates to 1 for image patches with bit values:
</p>
<p align="center">
  <img width="10%" src="https://github.com/olegranmo/blob/blob/master/y_1a.png">
</p>
and to 0 for other image patches.

#### A Tsetlin Automata Team for Composing Clauses
<p>
A Convolutional Tsetlin Machine consists of several clauses, how many are decided by the user. For each clause, a team of Tsetlin Automata decides which literals are to be included in the clause. There are one Tsetlin Automaton per literal, deciding whether the literal should be excluded or included in the clause:
<p>
<p align="center">
  <img width="90%" src="https://github.com/olegranmo/blob/blob/master/Example_Configuration_1a.png">
</p>
<p>
The team in the figure has for instance decided to include <img src="http://latex.codecogs.com/svg.latex?{x_{1,1}}, {x_{2,2}}, \lnot{x_{1,2}} \text{and} \lnot{x_{1,2}}" border="0"/>, producing the clause <img src="http://latex.codecogs.com/svg.latex?C = {x_{1,1}} \land {x_{2,2}} \land  \lnot{x_{1,2}} \land \lnot{x_{1,2}}" border="0"/>.
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
