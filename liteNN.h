
#include <string>
#pragma once
typedef struct {
	// weights
	float * wt;
	//hidden later to output layer weights
	float * x;
	//biases
	float * b;
	//hidden layer
	float * h;
	//otuput layer
	float * o;

	// number of biases
    int num_biases;
    // number of hidden layer neurons
    int num_hidden;
    //number of outputs 
    int num_outputs;
}
liteNN;


//Trains the liteNN with a single input and output with a learning rate and returns error of the NN
float train(liteNN, const float * inp, const float * arr, float learning_rate);
//function builds a liteNN object with given number of inputs, hidden neurons, and the number of output neurons
liteNN build(const int num_inp, const int num_hidden, const int num_out);
//returns the prediction done by the neural network
float * predict(liteNN, const float * inp);
//saves the liteNN todisk
float save(liteNN, char * path);
//laod saved liteNN instance from the disk
liteNN load(const char * path);
//remove a liteNN instance from the instance
void remove(liteNN);