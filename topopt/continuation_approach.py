# -*- coding: utf-8 -*-
"""
Continuation approach for SIMP

The penalty for SIMP gradually increases until it reaches the maximum penalty value.
 
"""
import numpy

def continuation_method(penalty_increment: float, penalty_max: int, 
                        loop_per_penalty: numpy.ndarray, numberOfLoops_penalty_max: int) -> numpy.ndarray:
    """
    Compute an array of penalty values when continuation approach is applied.

    Parameters
    ----------
    penalty_increment:
        The step size of penalty increment.
    penalty_max:
        The desired maximum penalty value for SIMP.
    loop_per_penalty:
        The iteration array to indicate the number of iterations for each penalty value.
    numberOfLoops_penalty_max:
        The number of iterations at maximum penalty value.
        
    Returns
    -------
    penalty_array:
        The penalty values for each step.
    loop_array:
        number of loops for each step.

    """
    
    # create penalty_array
    penalty_array = numpy.arange(1, penalty_max, penalty_increment )
    penalty_array = numpy.append(penalty_array, penalty_max)
    
    # number of penalty values
    numberOfPenalty = len(penalty_array)
    
    # create loop_array
    loop_array = numpy.ones(numberOfPenalty)*loop_per_penalty[-1]
    loop_array[0:len(loop_per_penalty)] = loop_per_penalty
    loop_array[-1] = numberOfLoops_penalty_max
 
    return penalty_array, loop_array, numberOfPenalty
    