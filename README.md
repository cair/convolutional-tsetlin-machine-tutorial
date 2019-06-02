# The Convolutional Tsetlin Machine (under construction)
The Convolutional Tsetlin Machine learns interpretable filters using propositional formulae. The propositional formulae are composed by a collective of Tsetlin Automata coordinated through a game. The Convolutional Tsetlin Machine is an interpretable alternative to convolutional neural networks.

## Learning Behaviour
The below figure depicts training and test accuracy for the Convolutional Tsetlin Machine on MNIST, epoch-by-epoch, in a single run. 
![Figure 4](https://github.com/olegranmo/blob/blob/master/performance_by_epoch_MNIST.png)
Test accuracy peaks at 99.50% after 168 epochs and 99.51% after 327 epochs. Further, it climbs quickly in the first epochs, passing 99% already in epoch 2. Training accuracy approaches 100%. 

## Tutorial on the Convolutional Tsetlin Machine

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

<p align="center">
  <img width="20%" src="https://github.com/olegranmo/blob/blob/master/Michael_Lvovitch_Tsetlin.png">
</p>
<p align="center">
  <b>Michael Lvovitch Tsetlin (22 September 1924 – 30 May 1966)</b>
</p>

<p>
The Convolutional Tsetlin Machine is based on the Tsetlin Automaton, introduced by M. L. Tsetlin in 1961. The Tsetlin Automaton is one of the pioneering solutions to the well-known multi-armed bandit problem and the first Finite State Learning Automaton.
</p>

#### Two-Action Tsetlin Automata
<p>
A two-action Tsetlin Automaton chooses among two actions, Action 1 or Action 2, and performs these sequentially in an environment. For each action performed, the environment responds stochastically with a Penalty or a Reward, according to an unknown reward probability distribution <img src="http://latex.codecogs.com/svg.latex?R=\[r_1, r_2\]" border="0"/>. When Action 1 is performed, the environment responds with a Reward with probability <img src="http://latex.codecogs.com/svg.latex?r_1" border="0"/>, otherwise, it responds with a Penalty. For Action 2, the probability of a Reward is <img src="http://latex.codecogs.com/svg.latex?r_2" border="0"/>.  By interacting with the environment, the goal of the Tsetlin Automaton is to, as quickly as possible, single in on the action that has the highest probability of eliciting a Reward.
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
The Tsetlin Automaton learns by changing state. Each state transition is decided by the feedback from the environment (Reward or Penalty). As shown in the figure above, a reward makes the Tsetlin Automaton change state away from the centre, while a penalty makes it change state towards the centre.
</p>

#### Example Run
<p>
The depicted Tsetlin Automaton is in state 3 (marked with a solid black circle). Accordingly, it selects Action 1. Assume this triggers a Penalty from the environment. The Tsetlin Automaton then moves from state 3 to state 4:
</p>

<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_2.png">
</p>
It is now in the right half of states and therefore selects Action 2. This time, the Tsetlin Automaton receives a Reward, updating its state accordingly:
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/fixed_deterministic_run_3.png">
</p>
<p>
At this point, it is quite confident that Action 2 is better than Action 1. Indeed, at least two consequtive Penalties are needed to make the Tsetlin Automaton change its mind and switch back to performing Action 1 again.
</p>

<p>
The above simple learning mechanism has some remarkable properties. It makes the Tsetlin Automaton act predictably, only changing action when switching between the two centre states. This supports stable collectives of many cooperating Tsetlin Automata, taking part in solving more complex problems. Further, the Tsetlin Automaton never stops learning. Therefore, it can adapt to changes in the environment. This helps avoiding getting stuck in local optima.  Finally, the accuracy and speed of learning is controlled by the number of states. By increasing the number of states and learning cycles towards infinity, the Tsetlin Automaton performs the optimal action with probability arbitrary close to unity. In other words, Tsetlin Automata learning is asymptotically optimal.
</p>

### The Architecture of the Convolutional Tsetlin Machine

#### Overview of Architecture
<p>
The structure of the classic Tsetlin Machine is shown below. This structure forms the core of the Convolutional Tsetlin Machine:
</p>
<p align="center">
  <img width="40%" src="https://github.com/olegranmo/blob/blob/master/Overall_Architecture.png">
</p>
<p>
In our 2D Noisy XOR example, the Convolutional Tsetlin Machine takes the propositional variables <img src="http://latex.codecogs.com/svg.latex?x_{1,1}, x_{2,1}, x_{1,2}, x_{2,2}" border="0"/> and their negations <img src="http://latex.codecogs.com/svg.latex?\lnot{x_{1,1}}, \lnot{x_{2,1}}, \lnot{x_{1,2}}, \lnot{x_{2,2}}" border="0"/> as input. These are referred to as literals, forming a bit vector. The input is fed to a fixed number of conjunctive clauses, how many is decided by the user. These clauses evaluate to either 0 or 1. Further, half of the clauses are assigned negative polarity (-), while the other half is assigned positive polarity (+). The polarity decides whether the clause output is negative or positive. Negative clauses are used to recognize class y=0, while positive clauses are used to recognize class y=1. Next, a summation operator aggregates the output of the clauses. The final classification is performed with a threshold function. Class y=1 is predicted if the number of positive clauses outputting 1 exceeds or equals the number of negative clauses outputting 1. Otherwise, class y=0 is predicted.
</p>

#### The Conjunctive Clause
<p>
A conjunctive clause is built by ANDing a selection of the available propositional variables <img src="http://latex.codecogs.com/svg.latex?x_{1,1}, x_{2,1}, x_{1,2}, x_{2,2}" border="0"/> and their negations <img src="http://latex.codecogs.com/svg.latex?\lnot{x_{1,1}}, \lnot{x_{2,1}}, \lnot{x_{1,2}}, \lnot{x_{2,2}}" border="0"/> . The clause <img src="http://latex.codecogs.com/svg.latex?C = {x_{1,1}} \land {x_{2,2}} \land  \lnot{x_{1,2}} \land \lnot{x_{1,2}}" border="0"/>, for instance, evaluates to 1 for image patches with bit values:
</p>
<p align="center">
  <img width="10%" src="https://github.com/olegranmo/blob/blob/master/y_1a.png">
</p>
and to 0 for other image patches.

#### The Tsetlin Automata Team for Composing Clauses
<p>
Each clause of the Convolutional Tsetlin Machine is composed by a team of Tsetlin Automata. These decide which literals are to be Excluded from the clause and which are to be Included. There is one Tsetlin Automaton per literal, deciding upon its inclusion:
<p>
<p align="center">
  <img width="90%" src="https://github.com/olegranmo/blob/blob/master/Example_Configuration_1a.png">
</p>
<p>
The team in the figure has for instance decided to include <img src="http://latex.codecogs.com/svg.latex?{x_{1,1}}, {x_{2,2}}, \lnot{x_{1,2}}" border="0"/> and <img src="http://latex.codecogs.com/svg.latex?\lnot{x_{1,2}}" border="0"/>, producing the clause <img src="http://latex.codecogs.com/svg.latex?C = {x_{1,1}} \land {x_{2,2}} \land \lnot{x_{1,2}} \land \lnot{x_{1,2}}" border="0"/>.
</p>

### Recognition with the Convolutional Tsetlin Machine

#### Clause Convolution Step
<p>
The Convolutional Tsetlin Machine recognizes patterns by first turning the input image into patches. For our 2D Noisy XOR example, the 3x3 input image is turned into four 2x2 patches:
</p>
<p align="center">
  <img width="60%" src="https://github.com/olegranmo/blob/blob/master/Convolution_Example.png">
</p>
<p>
Each conjunctive clause is then evaluated on each patch. For each clause, the outcome for each patch is ORed to produce the output of the clause. The figure shows this procedure for one of the clauses in our 2D Noisy XOR example.
</p>
<p>
To make the clauses location-aware, each patch is further enhanced with its coordinates within the image (see figure). Location awareness may prove useful in applications where both patterns and their location are distinguishing features, e.g. recognition of facial features such as eyes, eyebrows, nose, mouth, etc. in facial expression recognition. These coordinates are incorporated as additional propositional variables in the input vector. However, for the sake of simplicity, I will not considere the details of this incorporation here, and instead simply assume that the information on the coordinates already have been incorporated into the clauses.
</p>

#### Summation and Thresholding Step
<p>
The above clause is highlighted in the figure below, together with the seven other conjunctive clauses of our example configuration. Also, each clause has been annoted with the positional information it has incorporated, using thresholds on the x and y coordinates.
</p>
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/Recognition.png">
</p>
The above configuration consists of four positive clauses which represent XOR patterns that must be present in a positive example image (positive features) and four negative clauses which represent patterns that will not trigger a positive image classification (negative features). As explained earlier, the bit patterns inside each clause are represented by the output of eight Tsetlin Automata, one for each literal in the 2x2 filter. As in the classic Tsetlin Machine, the output from each clause is processed further by summation and then thresholding to decide the class.

### Learning with the Convolutional Tsetlin Machine

#### Allocation of Pattern Representation Resources

<p>
Each clause can be seen as a resource for representing patterns. With limited resources, it is critical to allocate the resources wisely. The Convolutional Tsetlin Machine seeks to allocate clauses uniformly among the crucial patterns in the dataset. This is achieved with a threshold T. That is, each time the outputs of the clauses are summed up, this threshold is the target value for the summation. For inputs of class y=0 the target value is -T and for inputs of class y=1 the target value is T. 
</p>
<p>The resources are allocated by controlling the intensity of the bandit learning feedback cycle. In brief, the feedback cycle is intensified the further away from the target value the clause output is, and comes to a complete standstill when the target value is reached. Let v denote the summed up clause output. Feedback intensity is modelled as the probability of activating each clause. For input of class y=0, the probability of activating a clause is:
<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\frac{T + \mathrm{max}(-T, \mathrm{min}(T, v))}{2T}" border="0"/>
</p>
<p>
and for input of class y=1, the clause activation probability is:
</p>
<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\frac{T - \mathrm{max}(-T, \mathrm{min}(T, v))}{2T}" border="0"/>
</p>
<p>
  <b>Remark.</b> A larger T (with a corresponding increase in the number of clauses) makes the learning more robust. This is because more clauses are involved in learning each specific pattern, introducing an ensemble effect.
</p>
<p>
Now, consider the Convolutional Tsetlin Machine configuration below. The Noisy 2D XOR problem has almost been solved. However, there is an imbalance in the representation of class y=1. Three clauses are allocated to represent one of the sub-patterns, while only a single clause has been allocated to the second pattern. 
</p>
<p align="center">
  <img width="105%" src="https://github.com/olegranmo/blob/blob/master/Learning.png">
</p>
<p>
For this example we use the threshold T=2. When an input image of the overrepresented pattern appears, nothing happens because T has been exceeded. However, when an input image of the underrepresented pattern comes along, as shown in the figure, the filter represented by the second positive clause matches the patch in the top-right corner of the image and is the only clause with output 1. similarly, none of the negative clauses respond since their patterns do not match the pattern found in the current patch.
</p>
<p>
Thus, the Convolutional Tsetlin Machine’s combined output is v=1. Learning of feature detectors proceeds as follows. With the threshold value set to T = 2, the probability of feedback is (T-v)/(2T)=0.25, and thus learning taking place, which pushes the output v towards T=2. This is achieved with what we refer to as Feedback Type I.
</p>

#### Feedback Type I
Type I feedback reinforces true positive output and reduces false negative output, that is, it makes y become 1 when it should be 1.
<p align="center">
  <img width="90%" src="https://github.com/olegranmo/blob/blob/master/Example_Configuration_4a.png">
</p>

<p align="center">
  <img width="90%" src="https://github.com/olegranmo/blob/blob/master/Example_Configuration_2a.png">
</p>

#### Feedback Type II

<p align="center">
  <img width="90%" src="https://github.com/olegranmo/blob/blob/master/Example_Configuration_3a.png">
</p>

### Goal State and Nash Equilibrium
A subsequent state of the CTM is shown below.
<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/Goal_State.png">
</p>
Note that there are now two positive clauses which detect their pattern in the top-right corner patch. The combined output of all clauses is 2; thus, no further learning is necessary for the detection of the XOR pattern in this patch. Also, the location of the occurrence of each pattern is included. The location information uses a bit representation as follows: Suppose an XOR pattern occurs at the three X-coordinates 1, 4, and 6. For the corresponding binary location representation, these coordinates are considered thresholds: If a coordinate is greater than a threshold, then the corresponding bit in the binary representation will be 0; otherwise, it is set to 1. Thus, the representation of the X-coordinates 1, 4, and 6 will be ‘111’, ‘011’ and ‘001’, respectively. These representations of the location of 2x2 patterns are also learned by TAs.

## Demo

## Citation

```bash
@article{granmo2019convtsetlin,
  author = {{Granmo}, Ole-Christoffer and {Glimsdal}, Sondre and {Jiao}, Lei and {Goodwin}, Morten and {Omlin}, Christian W. and {Berge}, Geir Thore},
  title = "{The Convolutional Tsetlin Machine}",
  journal={arXiv preprint arXiv:1905.09688}, year={2019}
}
```

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
