#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("WeightedAverage")
    .Input("noisy_img: float32")
    .Input("weights: float32")
    .Output("averaged: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        ::tensorflow::shape_inference::ShapeHandle output;

        // Set the output to img_w x img_h x 3 (RGB)
        TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(1), 3, c->MakeDim(3), &output));

        c->set_output(0, output);
        return Status::OK();
    });


class WeightedAverageOp : public OpKernel {
    public:
        explicit WeightedAverageOp(OpKernelConstruction * context) : OpKernel(context) {}

        void Compute(OpKernelContext * context) override {
            
            // Get the noisy image tensor
            const Tensor & noisy_img_tensor = context->input(0);
            OP_REQUIRES(context, noisy_img_tensor.shape().dims() == 4,
                    errors::InvalidArgument("Image should be a 4D tensor"));
            auto noisy_img = noisy_img_tensor.tensor<float, 4>();

            // Get the image dimensions
            const int batch_size = noisy_img.dimension(0);
            const int noisy_img_width = noisy_img.dimension(1);
            const int noisy_img_height = noisy_img.dimension(2);

            OP_REQUIRES(context, noisy_img.dimension(3) == 3,
		    errors::InvalidArgument("Images should be RGB (have 3 dimensions)."));

            // Get the weights tensor
            const Tensor & weights_tensor = context->input(1);
            OP_REQUIRES(context, weights_tensor.shape().dims() == 4,
                    errors::InvalidArgument("Weights should be a 4D tensor"));
            auto weights = weights_tensor.tensor<float, 4>();

            // Get the information about the kernel
            const int kernel_area = weights.dimension(3);
            const int kernel_size = std::sqrt(kernel_area);
            const int padding = (kernel_size - 1) / 2;
            const int weights_width = noisy_img_width - 2 * padding;
            const int weights_height = noisy_img_height - 2 * padding;

	    OP_REQUIRES(context, weights.dimension(0) == batch_size,
		    errors::InvalidArgument("Images and weights should have the same number of batches."));

	    OP_REQUIRES(context, kernel_size % 2 == 1,
		    errors::InvalidArgument("Kernel size must be odd."));

	    OP_REQUIRES(context, kernel_area == kernel_size * kernel_size,
		    errors::InvalidArgument("Kernel area must be the square of an odd number."));

            OP_REQUIRES(context, weights.dimension(1) == weights_width && weights.dimension(2) == weights_height,
		    errors::InvalidArgument("Images and weights should have the same image size minus the chop-off-border."));

            // Create the output (prediction) tensor
            Tensor * prediction_img_tensor = NULL;
            OP_REQUIRES_OK(
                context, 
                context->allocate_output(
                    0, 
                    {batch_size, weights_width, weights_height, 3},
                    &prediction_img_tensor
                )
            );

            auto prediction_img = prediction_img_tensor->tensor<float, 4>().setConstant(0);

            #pragma omp parallel for
            for (int batch = 0 ; batch < batch_size ; batch++) {
                for (int i = 0 ; i < weights_width ; i++) {
                    for (int j = 0 ; j < weights_height ; j++) {
                        for (int k1 = 0 ; k1 < kernel_size ; k1++) {
                            for (int k2 = 0 ; k2 < kernel_size ; k2++) {
                                const int index_in_patch = kernel_size * k1 + k2; 
                                const float weight = weights(batch, i, j, index_in_patch);
                                prediction_img(batch, i, j, 0) += weight * noisy_img(batch, i + k1, j + k2, 0);
                                prediction_img(batch, i, j, 1) += weight * noisy_img(batch, i + k1, j + k2, 1);
                                prediction_img(batch, i, j, 2) += weight * noisy_img(batch, i + k1, j + k2, 2);
                            }
                        }
                    }
                }   
            }
        }
};

REGISTER_KERNEL_BUILDER(
    Name("WeightedAverage").Device(DEVICE_CPU), 
    WeightedAverageOp
);

REGISTER_OP("WeightedAverageGradients")
    .Input("weights: float32")
    .Input("input_grads: float32")
    .Input("input_colours: float32")
    .Output("output_grads: float32");

class WeightedAverageGradientsOp : public OpKernel {

    public:
        explicit WeightedAverageGradientsOp(OpKernelConstruction * context) : OpKernel(context) {}
        
        void Compute(OpKernelContext * context) override {

            // Get the weights
            const Tensor & weights_tensor = context->input(0);
            OP_REQUIRES(context, weights_tensor.shape().dims() == 4,
                errors::InvalidArgument("Weights should be a 4D tensor")
            );
            auto weights = weights_tensor.tensor<float, 4>();

            // Get the gradients
            const Tensor & grads_tensor = context->input(1);
            auto grads = grads_tensor.tensor<float, 4>();

            // Get the input colours
            const Tensor & colours_tensor = context->input(2);
            auto colours = colours_tensor.tensor<float, 4>();

            Tensor * output_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    weights_tensor.shape(),
                    &output_tensor
                )
            );
            auto output = output_tensor->tensor<float, 4>();

            const int batch_size = weights.dimension(0);
            const int img_width = weights.dimension(1);
            const int img_height = weights.dimension(2);
            const int kernel_area = weights.dimension(3);
            const int kernel_size = std::sqrt(kernel_area);

            #pragma omp parallel for
            for (int batch = 0 ; batch < batch_size ; batch++) {
                for (int i = 0 ; i < img_width ; i++) {
                    for (int j = 0 ; j < img_height ; j++) {
                        for (int k1 = 0 ; k1 < kernel_size ; k1++) {
                            for (int k2 = 0 ; k2 < kernel_size ; k2++) {
				const int patch_id = kernel_size * k1 + k2;
				output(batch, i, j, patch_id) = grads(batch, i, j, 0) * colours(batch, i + k1, j + k2, 0) +
                					        grads(batch, i, j, 1) * colours(batch, i + k1, j + k2, 1) +
                					        grads(batch, i, j, 2) * colours(batch, i + k1, j + k2, 2);
                            }
                        }
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(
    Name("WeightedAverageGradients").Device(DEVICE_CPU),
    WeightedAverageGradientsOp
);
