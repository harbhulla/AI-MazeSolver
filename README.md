# AI-MazeSolver

<h2>Research Papers for DQN and DDQN</h2>
<h3> DQN - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf</h3>
<h3> DDQN - https://arxiv.org/abs/1509.06461</h3>

<h2>Maze Introduction</h2>
<p>This maze is a third party environment and consists of several different maze sizes. For the sake of simplicity and focusing on the algorithms and neural network itself, I decided to start off with a 5x5 maze and later ambitiously increased the size of the grid to 10x10 and got very unexpected results. For every timestep, you lose -0.001 and everytime the agent gets to the end it receives a +1. Overtime, it learns to go through the maze faster and faster.</p>

<h2>Neural Network for 5x5 Maze</h2>
<p> For optimial results, the neural network consists of just 1 hidden layer, node size being 1024 x 1024.</p>

<h2>Hyperparameters</h2>
<p> Through trial and error on a DDQN, I realized the best and most consistent results come with these hyperparameters.</p>
<p>GAMMA - 0.999<br>
Greedy Strategy - 1, 0.999 for decay - 0.01 for min<br>
Batch Size - 32<br>
Every 500 timesteps I updated my target network </p>


<h1>5x5 Maze AI</h1>
<p> Since it's just a 5x5 Maze, exploration is rather quick and soon enough with enough random actions the AI gets to the end and gets that reward of +1. Moreover interestingly enough, although the maze is just 5x5 compared to the cartpole and rover this took significantly more time in order to solve than those two. Each episode was about 5-7 minutes long and it took 160 episodes in order to get a consistent and efficent result.</p>

![startMaze](https://user-images.githubusercontent.com/41172710/183312364-432fa301-1b6d-4668-b646-19d7263f1c15.gif)
<br>
<p>One of the most basic and fundamental concepts in artificial intelligence is the concept of local and global minimia derived from calculus 3 e.g gradient descent.</p>

![1_ZC9qItK9wI0F6BwSVYMQGg](https://user-images.githubusercontent.com/41172710/183312466-a919977c-80b7-47e3-af13-60ce6652e536.png)<br>

<p>Gradient descent is an optimization algorithm and the backbone of AI. Since I just picked up AI this summer and I understood the concept of gradient descent, but never saw it in action until now. In this basic 5x5 maze, when I wasn't using the best hyperparameters or neural network structure, it would often get stuck in the local minimia. Local minimia is something we want to avoid, since it's not the global minimia and we want better results. Referring back to the previous image, we start at the top of the hill and slowly make our way down as the neural network minimizes the mean squared error and our goal is to the get to the global minimia. Eventually though, after 150+ episodes, the AI started to perform consistently as shown below.</p>

![endMaze](https://user-images.githubusercontent.com/41172710/183312625-3abb30be-0ca6-43a0-9b81-cb60b256451e.gif)
<br>

<h2>10x10 Maze Adventure</h2>

<p>Although I succeeded in solving a 5x5, I want to push the algorithm further and attempt to solve a 10x10 maze. Intuitively, one might think, "it's just a small jump from a 5x5 to a 10x10" and that is something I thought as well. However, I was completely wrong. The jump from 5x5 to 10x10 was significant. It took FAR longer in order for my algorithm to converge compared to the 5x5 one. To put it into perspective, the 5x5 maze took the agent maybe ~2 hours to solve, but the 10x10 maze would take DAYS. </p>

![10x10maze](https://user-images.githubusercontent.com/41172710/183313577-3525011f-5581-472e-935b-aae501278f58.gif)
<br>

<p>Unfortunately, me not having the computational power in order to solve this maze and run it for more episodes, the best results i got were 10 episodes after 3 hours of training of it just completing the maze, not having an efficient pathway</p><br>
<p>This test shows the massive difference between the large amount of computations a neural network with 4096 nodes and 2 hidden layers has to do in such a short period of time. Something that to us humans may seem basic, under the hood, millions of computations are happening every minute, showcasing the power of algorithms and computers</p>

