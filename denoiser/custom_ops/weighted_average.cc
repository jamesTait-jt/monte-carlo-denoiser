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

            // Convert the output to a flat tensor
            auto prediction_flat = prediction_img->flat<float>();

            // Apply the weights to the neighbourhood
            const int N = prediction_flat.size();
                
            TensorShape weights_shape = weights.shape();
            TensorShape img_shape_with_padding = noisy_img.shape();

            DCHECK_EQ(img_shape_with_padding.dim_size(2), 3);

            // Ensure the kernel is square
            DCHECK_EQ(
                weights_shape.dim_size(2), 
                weights_shape.dim_size(3),
            );

            const int kernel_radius = weights_shape.dim_size(2) / 2;

            DCHECK_EQ(kernel_radius, 10);

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
