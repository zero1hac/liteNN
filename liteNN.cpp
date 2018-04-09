#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include "liteNN.h"


using namespace std;


//utility functions
float max(const float a, const float b){
	return a>b ? a : b;
}

//Actiavtion functions
float relu(const float x){
	return max(0, x);
}

float leaky_relu(const float x){
	return x>0 ? x : 0.01*x;
}

float sigmoid(const float x){
	return 1.0 / (1.0 + expf(x));
}


//Partial derivatives of the Actiavtion functions
float pdrelu(const float x){
	return x>0 ? 1 : 0;
}

float pdleaky_relu(const float x){
	return x>0 ? 1 : 0.01;
}

float pdsigmoid(const float x){
	return x * (1.0 - x);
}

//Generate random floating point number from 0.0 to 1.0
float randf(){
	return rand()/(float) RAND_MAX;
}

//Error function
float errfn(const float a, const float b){
	return 0.5 * powf(a - b, 2.0);
}

//Partial derivative of the loss function
float pderrorf(const float a, const float b){
	return a - b;
}

//Returns total error
float sum_error(const float * const arr, const float * const tar, const int sz){
	float sum = 0.0;
	for(int i=0; i<sz; i++){
		sum += errfn(arr[i], tar[i]);
	}
	return sum;
}

void backprop(const liteNN net, const float * const inp, const float * const arr, float learning_rate){
	
}
int main(){
	srand(time(NULL));
	cout<<relu(5.44)<<endl;
	cout<<leaky_relu(-5.44)<<endl;
	cout<<sigmoid(10.01)<<endl;
	cout<<relu(-90.0);
	cout<<pdrelu(5.44)<<endl;
	cout<<pdleaky_relu(-5.44)<<endl;
	cout<<pdsigmoid(10.01)<<endl;
	cout<<pdrelu(-90.0)<<endl;
	cout<<randf();
	return 0;
}