import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	arr = np.array(gradient)
	magnitude = np.linalg.norm(arr)
	if magnitude == 0:
		direction = [0.0] * len(gradient)
		descent_direction = [0.0] * len(gradient)
	else:
		direction = (arr / magnitude)
		descent_direction = (-1 * direction)
	outdic = {
		"magnitude" : magnitude,
		"direction" : direction,
		"descent_direction" : descent_direction
	}
	return outdic
