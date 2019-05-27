# The Convolutional Tsetlin Machine (coming soon)
The Convolutional Tsetlin Machine learns interpretable filters using propositional formulae. It is an interpretable alternative to convolutional neural networks.

## Step-by-Step Walkthrough of Inference

### Recognition

<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/Recognition.png">
</p>

### Learning

<p align="center">
  <img width="105%" src="https://github.com/olegranmo/blob/blob/master/Learning.png">
</p>

### Goal State

<p align="center">
  <img width="65%" src="https://github.com/olegranmo/blob/blob/master/Goal_State.png">
</p>

## Learning Behaviour
The below figure depicts training and test accuracy for the Convolutional Tsetlin Machine on MNIST, epoch-by-epoch, in a single run. 
![Figure 4](https://github.com/olegranmo/blob/blob/master/performance_by_epoch_MNIST.png)
Test accuracy peaks at 99.50% after 168 epochs and 99.51% after 327 epochs. Further, test accuracy climbs quickly in the first epochs, passing 99% already in epoch 2. Training accuracy approaches 100%. 

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
