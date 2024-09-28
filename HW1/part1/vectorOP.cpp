#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
          
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }             
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  	//
  	// PP STUDENTS TODO: Implement your vectorized version of
  	// clampedExpSerial() here.
  	//
  	// Your solution should work for any value of
  	// N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  	//
	__pp_vec_float x; 
    __pp_vec_int ex;
	__pp_vec_float result;


    __pp_vec_int zerosInt = _pp_vset_int(0);
    __pp_vec_int onesInt  = _pp_vset_int(1);

    __pp_vec_float ninesFloat = _pp_vset_float(9.999999f);

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        int width = i + VECTOR_WIDTH <= N ? VECTOR_WIDTH : N - i; 
    
        __pp_mask maskAll  = _pp_init_ones(width);

    	_pp_vload_float(x, values + i, maskAll);
        _pp_vload_int(ex, exponents + i, maskAll);

        result = _pp_vset_float(1.f);

		
        __pp_mask zeroIndexEx;

        _pp_veq_int(zeroIndexEx, ex, zerosInt, maskAll);

        __pp_mask copyResult = _pp_mask_not(zeroIndexEx);

        int countDone = _pp_cntbits(zeroIndexEx);

        while (countDone < VECTOR_WIDTH) {
            __pp_mask doCalculation = _pp_mask_not(zeroIndexEx);
            _pp_vmult_float(result, result, x, doCalculation);
            _pp_vsub_int(ex, ex, onesInt, doCalculation);

            _pp_veq_int(zeroIndexEx, ex, zerosInt, maskAll);
            countDone = _pp_cntbits(zeroIndexEx);
        }
        
        __pp_mask largerThanNines; 
        _pp_vgt_float(largerThanNines, result, ninesFloat, copyResult);
        largerThanNines = _pp_mask_and(largerThanNines, copyResult);
        _pp_vset_float(result, 9.999999f, largerThanNines);

        _pp_vstore_float(output + i, result, maskAll);
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //

    __pp_vec_float sum = _pp_vset_float(0.0f);
    __pp_vec_float x;

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        int width = i + VECTOR_WIDTH <= N ? VECTOR_WIDTH : N - i; 

        __pp_mask maskAll  = _pp_init_ones(width);

        _pp_vload_float(x, values + i, maskAll);

        _pp_vadd_float(sum, sum, x, maskAll);
    }

    float counter = VECTOR_WIDTH;

    while (counter > 1) {
        _pp_hadd_float(sum, sum);
        _pp_interleave_float(sum, sum);
        counter /= 2;
    }

    return sum.value[0];
}