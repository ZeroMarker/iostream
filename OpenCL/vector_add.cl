__kernel void vector_add(__global const float *input_x,
	__global const float *input_y,
	__global float *output)
{
	int gid = get_global_id(0);
	output[gid] = input_x[gid] + input_y[gid];
}