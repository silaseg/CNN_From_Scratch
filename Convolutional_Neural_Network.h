#ifndef __CONVOLUTIONAL_NEURAL_NETWORK__
#define __CONVOLUTIONAL_NEURAL_NETWORK__

void reset_block(float ***mem);
void find_label(FILE *file, int index_max_pred);
void reset_block_dense(float *mem);
void init_image();
void init_convolution_weight();
void init_dense_weight();
void init_blocks_dense();
void initialization();
void free_image();
void free_convolution_weights();
void free_dense_weights();
void free_blocks();
void free_dense_blocks();
void free_all();
void read_weights_file(char *in_file, int levels);
void read_image_file(char *in_file);
void image_normalization();
void convolution(float **matrix, float **kernel, float **out, int size);
void bias_and_relu(float **out, float bs, int size);
void bias_and_relu_flatten(float *out, float *bs, int size, int relu);
float maxpooling_2x2(float a, float b, float c, float d);
void maxpooling(float **out, int size);
void flatten(float ***in, float *out, int shape0, int shape1, int shape2);
void dense(float *in, float **weights, float *out, int shape_in, int shape_out);
void softmax(float *out, int shape_out);
int save_dense_to_file(float *mem, int shape0);
int VGG16();
char *trimwhitespace(char *str);

#endif