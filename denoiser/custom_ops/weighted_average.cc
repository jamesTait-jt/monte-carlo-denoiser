#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("WeightedAverage")
    .Input("noisy_img: float32")
    .Input("weights: float32")
    .Output("averaged: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext * c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


class WeightedAverageOp : public OpKernel {
    public:
        explicit WeightedAverageOp(OpKernelConstruction * context) : OpKernel(context) {}

        void Compute(OpKernelContext * context) override {
            // Get the noisy image tensor
            const Tensor & noisy_img = context->input(0);

            // Get the weights tensor
            const Tensor & weights = context->input(1);

            // Create the output (prediction) tensor
            Tensor * prediction_img = NULL;
            OP_REQUIRES_OK(
                context, 
                context->allocate_output(
                    0, 
                    noisy_img.shape(),
                    &prediction_img
                )
            );

            // Get the shapes of the input tensors
            TensorShape weights_shape = weights.shape();
            TensorShape img_shape_with_padding = noisy_img.shape();

            // Get the batch size and ensure it's the same for each one
            const int batch_size_weights = weights_shape.dim_size(0);
            const int batch_size_img = weights_shape.dim_size(0);
            DCHECK_EQ(batch_size_weights, batch_size_img);
            const int batch_size = batch_size_img;

            // Ensure it's an RGB image
            DCHECK_EQ(img_shape_with_padding.dim_size(3), 3);

            // Ensure the kernel is square
            DCHECK_EQ(
                weights_shape.dim_size(3), 
                weights_shape.dim_size(4)
            );
    
            // Ensure the kernel has odd dimension
            DCHECK_EQ(weights_shape.dim_size(3) % 2, 1);

            // Get the image/weights dimensions
            const int padded_img_width = img_shape_with_padding.dim_size(1);
            const int padded_img_height = img_shape_with_padding.dim_size(2);
            const int kernel_width = weights_shape.dim_size(3);
            const int kernel_height = weights_shape.dim_size(4);

            // Ensure the image is square
            DCHECK_EQ(padded_img_width, padded_img_height);

            // Ensure the kernel is smaller than the padded image
            DCHECK(weights_shape.dim_size(3) < padded_img_width);

            // Ensure the kernel is square
            DCHECK_EQ(kernel_width, kernel_height);

            // Get the radius of the kernel
            const int kernel_radius = std::floor(kernel_width / 2);

            // Get the accessible version of the noisy image tensor
            auto noisy_tensor = noisy_img.tensor<float, 4>();

            // Get the accessible version of the output tensor
            auto prediction_tensor = prediction_img->tensor<float, 4>();

            // Get the accessible version of the weights tensor
            auto weights_tensor = weights.tensor<float, 5>();

            for (int batch = 0 ; batch < batch_size ; batch++) {
                for (int i = 0 ; i < padded_img_height ; i++) {
                    for (int j = 0 ; j < padded_img_width ; j++) {
                        // If we are in the padding - just set it to zero
                        if ((i < kernel_radius) || (i + kernel_radius >= padded_img_height) ||
                           (j < kernel_radius) || (j + kernel_radius >= padded_img_width)) {
                            prediction_tensor(batch, i, j, 0) = 0;
                            prediction_tensor(batch, i, j, 1) = 0;
                            prediction_tensor(batch, i, j, 2) = 0;
                        // Otherwise, calculate the weighted average of the noisy
                        // neighbourhood
                        } else {
                            float sum_r = 0;
                            float sum_g = 0;
                            float sum_b = 0;
                            for (int k1 = -kernel_radius ; k1 <= kernel_radius ; k1++) {
                                for (int k2 = -kernel_radius ; k2 <= kernel_radius ; k2++) {
                                    float weight_value = weights_tensor(batch, i, j, k1, k2);
                                    float neighbour_value_r = noisy_tensor(batch, k1 + i, k2 + j, 0);
                                    float neighbour_value_g = noisy_tensor(batch, k1 + i, k2 + j, 1);
                                    float neighbour_value_b = noisy_tensor(batch, k1 + i, k2 + j, 2);
                                    sum_r += neighbour_value_r * weight_value;
                                    sum_g += neighbour_value_g * weight_value;
                                    sum_b += neighbour_value_b * weight_value;
                                    }
                            }
                            prediction_tensor(batch, i, j, 0) = sum_r;
                            prediction_tensor(batch, i, j, 1) = sum_g;
                            prediction_tensor(batch, i, j, 2) = sum_b;
                        }
                    }
                }   
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("WeightedAverage").Device(DEVICE_CPU), WeightedAverageOp);

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};


REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
