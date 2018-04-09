# Emory-2018-ml-project-3 (End April 15, 2018)
A joint project for ML course

# Tasks:

## Tune paramters

1. batch_size:  A too large
batch_size will increase the computation load for every backpropogation and a too small batch_size will make the gradients noisy and influence the convergence rate of the optimization algorithm

2. conv-net: Try different structures of the conv-net and compare the performance with each other. The components you can modify but not limited to are: 

   * initial learning rate
   *  size of the filters
   * number of filters
   * number of conv layers
   * number of pooling layers
   *  the ways of paddings
   * the choices of activation functions

3. built-in optimizers: [understand the mathematical mechanisms](https://www.tensorflow.org/api_guides/python/train#Optimizers) and  observe the convergence rate compared to steepest gradient descent

   * steepest gradient descent (default)
   * momentum optimizer
   * adam optimizer

## Tensorboard:
 Add tensorboard part to your model. Tensorboard is a powerful tool, which helps you monitor how your training process goes on. It is a good way to visualize your model (computation graph you generated with tensorflow) and help you tune a good combination of hyper-parameters. You can even visualize how the weights of your model change with respect to training epochs. Read the tutorial below and watch the great video on this page for more information on adding tensorboard to your code.

(https://www.tensorflow.org/programmers_guide/summaries_and_tensor board)
Hint: In this project, you may not need to initialize tf.summary.FileWriter by your own since tf.estimator.Estimator will automatically help you write the summary you want to record into the event file (See this discussion). You may just add tf.summary.scalar('loss', loss) after the loss computed in the original code and then you can view the result in the tensorboard after the training finishes.
To visualize the tensorboard event file on the browser:

` Tensorboard --logdir path_to_your_model_dir --port=8008`

 Then type `http://localhost:8008`  in your the browser


## Write up report

Write up a report about how you did the experiment in step 1) – 4). Report
the performance on testing data based on the best model you have tuned (the structure of the model, the regularization methods you applied to prevent overfitting and any other components you added to make your model better and etc.) and attach the training plots generated by tensorboard to show how the training process goes. You may need to include how training and validation loss changes, how training and validation accuracy changes and the computation graph you generated (in the GRAPH part of the tensorboard log file).

# TO-DO list
[] Randomize: Divide training data into 80/20 traning/validation data.

1. Write a full training & evaluation code

2. Tune hyperprameters

3. Write report 


# Group member

* Cai
* Ren
* Wang
