#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <cmath>
#include <iostream>

REGISTER_OP("WeightedAverage")
	.Input("images: float")
	.Input("weights: float")
	.Output("filtered: float");

using namespace tensorflow;

class WeightedAverageOp : public OpKernel {
public:
	explicit WeightedAverageOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override {

		const Tensor &weight_tensor = context->input(1);
		OP_REQUIRES(context, weight_tensor.shape().dims() == 4,
		        errors::InvalidArgument("Images as well as weights should be 4-D tensors."));
		auto weights = weight_tensor.tensor<float, 4>();

		const int nbatches    = weights.dimension(0);
		const int kernel_area = weights.dimension(3);
		const int kernel_size = sqrt(kernel_area);
		const int border      = (kernel_size - 1)/2;


		OP_REQUIRES(context, kernel_size % 2 == 1,
		        errors::InvalidArgument("Kernel size must be odd."));

		OP_REQUIRES(context, kernel_area == kernel_size * kernel_size,
		        errors::InvalidArgument("Kernel area must be the square of an odd number."));


		const Tensor &image_tensor = context->input(0);
		OP_REQUIRES(context, image_tensor.shape().dims() == 4,
		        errors::InvalidArgument("Images as well as weights should be 4-D tensors."));
		auto images = image_tensor.tensor<float, 4>();

		const int imagew = images.dimension(1);
		const int imageh = images.dimension(2);
		const int weightw = imagew - 2 * border;
		const int weighth = imageh - 2 * border;

		OP_REQUIRES(context, images.dimension(0) == nbatches,
		        errors::InvalidArgument("Images and weights should have the same number of batches."));
		OP_REQUIRES(context, weights.dimension(1) == weightw &&
							 weights.dimension(2) == weighth,
		        errors::InvalidArgument("Images and weights should have the same image size minus the chop-off-border."));
		OP_REQUIRES(context, weights.dimension(1) == weightw &&
							 weights.dimension(2) == weighth,
		        errors::InvalidArgument("Images and weights should have the same image size minus the chop-off-border."));
		OP_REQUIRES(context, images.dimension(3) == 3,
		        errors::InvalidArgument("Images should be RGB (have 3 dimensions)."));



		Tensor *output_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nbatches, weightw, weighth, 3}, &output_tensor));
		auto output = output_tensor->tensor<float, 4>().setConstant(0);

        #pragma omp parallel for
		for (int b = 0; b < nbatches; ++b)
			for (int x = 0; x < weightw; ++x)
				for (int y = 0; y < weighth; ++y)
					for (int patchx = 0; patchx < kernel_size; ++patchx)
						for (int patchy = 0; patchy < kernel_size; ++patchy) {
							const int index_in_patch = kernel_size*patchx + patchy;
							const float w = weights(b,x,y,index_in_patch);
							output(b,x,y,0) += w * images(b,x+patchx,y+patchy,0);
							output(b,x,y,1) += w * images(b,x+patchx,y+patchy,1);
							output(b,x,y,2) += w * images(b,x+patchx,y+patchy,2);
						}



	}
};

REGISTER_KERNEL_BUILDER(
	Name("WeightedAverage").Device(DEVICE_CPU),
	WeightedAverageOp
);


REGISTER_OP("WeightedAverageGradients")
	.Input("weights: float")
	.Input("input_grads: float")
	.Input("input_colors: float")
	.Output("output_grads: float");

class WeightedAverageGradientsOp : public OpKernel {
public:
	explicit WeightedAverageGradientsOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override {
		// Get weights
		const Tensor &weight_tensor = context->input(0);
		OP_REQUIRES(context, weight_tensor.shape().dims() == 4,
		        errors::InvalidArgument("weights should be a 4-D tensor."));
		auto weights = weight_tensor.tensor<float, 4>();

		// Get grads
		const Tensor &grads_tensor = context->input(1);
		auto grads = grads_tensor.tensor<float, 4>();

		// Get input colors
		const Tensor &colors_tensor = context->input(2);
		auto colors = colors_tensor.tensor<float, 4>();

		Tensor *output_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, weight_tensor.shape(), &output_tensor));
		auto output = output_tensor->tensor<float, 4>();

		const int nbatches    = weights.dimension(0);
		const int w 	      = weights.dimension(1);
		const int h           = weights.dimension(2);
		const int kernel_area = weights.dimension(3);
		const int kernel_size = sqrt(kernel_area);
		// const int border      = (kernel_size - 1)/2;

        #pragma omp parallel for
		for (int b = 0; b < nbatches; ++b)
			for (int x = 0; x < w; ++x)
				for (int y = 0; y < h; ++y)
					for (int patchx = 0; patchx < kernel_size; ++patchx)
						for (int patchy = 0; patchy < kernel_size; ++patchy) {
							const int patch_id = kernel_size*patchx + patchy;
							output(b,x,y,patch_id) = grads(b, x, y, 0) * colors(b, x+patchx, y+patchy, 0) +
            													 grads(b, x, y, 1) * colors(b, x+patchx, y+patchy, 1) +
            													 grads(b, x, y, 2) * colors(b, x+patchx, y+patchy, 2);
						}



	}
};

REGISTER_KERNEL_BUILDER(
	Name("WeightedAverageGradients").Device(DEVICE_CPU),
	WeightedAverageGradientsOp
);
