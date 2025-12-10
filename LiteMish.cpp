#include <cmath>
#include <random>
#include "esp_timer.h"

unsigned long start = 0; // Start time (microseconds)
unsigned long stop  = 0; // Stop  time (microseconds)

const size_t N = 1000; // Number of scalar calls
float L[N] = {}; // Array to store random numbers

//----------------------------------------------------------------- Constant parameters
#define x1 -7.422187f
#define x2 -0.737440f
#define a -0.0074315f
#define b +0.7453866f

//------------------------------------------------------ C++ implementation of LiteMish
static inline float LiteMish(float x) {

    if (x > 0.0f){
      return x;
    }else if (x > x2){
      return b*x*x + x;
    }else if (x > x1){
      return a*(x-x1)*(x-x1);
    }else{
      return 0.0f;
    }
}

//--------------------------------------------------------------------------------
static inline float Mish(float x){ return x*tanh(Softplus(x)); }

//--------------------------------------------------------------------------------
static inline float Swish(float x){ return x/(1.0f + exp(-x)); }

//--------------------------------------------------------------------------------
static inline float ELU(float x){ return  (x > 0.0f) ? (x) : (exp(x) - 1.0f);}

//--------------------------------------------------------------------------------
static inline float Softplus(float x){ return log(1.0f + exp(x)); }

//--------------------------------------------------------------------------------
static inline float Squareplus(float x){ return 0.5f*(x + sqrt(x*x + 4.0f)); }



void setup() {
  // This function will be executed once

  unsigned int seed = 0;

  std::default_random_engine generator(seed);
  float mean = 0.0;
  float std_dev = 5.0;
  std::normal_distribution<float> distribution(mean, std_dev);

  /* 
  Sampling random numbers from a normal distribution
  and storing them in L
  */

  for(size_t i = 0; i < N; i++){
    L[i] = distribution(generator);
  }

  Serial.begin(9600);
}

volatile float var = 0.0;

void loop() {
  
  start = esp_timer_get_time();

  for(size_t i = 0; i < N; i++){
    var = LiteMish(L[i]);
  }

  stop = esp_timer_get_time();

  Serial.println(stop - start);
}
