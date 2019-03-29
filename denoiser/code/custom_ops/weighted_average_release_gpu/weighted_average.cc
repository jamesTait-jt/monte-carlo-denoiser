#include <cmath>
#include <iostream>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;
using namespace shape_inference;

REGISTER_OP("WeightedAverage")
  .Input("images: float")
  .Input("weights: float")
  .Output("filtered: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ShapeHandle images_shape = c->input(0);
    ShapeHandle weights_shape = c->input(1);
    std::vector<DimensionHandle> dims;
    dims.reserve(4);
    dims.push_back(c->Dim(images_shape, 0));
    dims.push_back(c->Dim(weights_shape, 1));
    dims.push_back(c->Dim(weights_shape, 2));
    dims.push_back(c->Dim(images_shape, 3));
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  });

class WeightedAverageOp : public OpKernel {
public:
  explicit WeightedAverageOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input 'images' tensor.
    const Tensor &images_tensor = context->input(0);
    OP_REQUIRES(context, images_tensor.shape().dims() == 4,
                errors::InvalidArgument("The input 'images' is not a 4-D tensor."));
    auto images = images_tensor.tensor<float, 4>();

    // Grab the input 'weights' tensor.
    const Tensor &weights_tensor = context->input(1);
    OP_REQUIRES(context, weights_tensor.shape().dims() == 4,
                errors::InvalidArgument("The input 'weights' is not a 4-D tensor."));
    auto weights = weights_tensor.tensor<float, 4>();

    const int nbatches    = weights.dimension(0);
    const int kernel_area = weights.dimension(3);
    const int kernel_size = sqrt(kernel_area);

    // Ensure the 'images' and 'weights' inputs are consistent.
    OP_REQUIRES(context, images.dimension(0) == nbatches,
                errors::InvalidArgument("Images and weights should have the same number of batches."));
    OP_REQUIRES(context, images.dimension(3) == 3,
                errors::InvalidArgument("Images should be RGB (have 3 dimensions)."));

    // Ensure the kernel is the a square with an odd-sized side.
    OP_REQUIRES(context, kernel_size % 2 == 1,
                errors::InvalidArgument("Kernel size must be odd."));
    OP_REQUIRES(context, kernel_area == kernel_size * kernel_size,
                errors::InvalidArgument("Kernel area must be square of an odd number."));

    // The image buffer is assumed to be padded with a border equal to the
    // kernel radius.
    const int border     = (kernel_size - 1)/2;
    const int img_width  = images.dimension(1);
    const int img_height = images.dimension(2);
    const int wgt_width  = img_width  - 2 * border;
    const int wgt_height = img_height - 2 * border;

    // Ensure the image buffer is properly padded.
    OP_REQUIRES(context, weights.dimension(1) == wgt_width &&
                weights.dimension(2) == wgt_height,
                errors::InvalidArgument("Images and weights should have the same image size minus the chop-off-border."));

    // Create the output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {nbatches, wgt_width, wgt_height, 3}, &output_tensor));
    auto output = output_tensor->tensor<float, 4>().setConstant(0);

    // Perform the weighted averaging
    #pragma omp parallel for
    for (int b = 0; b < nbatches; ++b)
      for (int x = 0; x < wgt_width; ++x)
        for (int y = 0; y < wgt_height; ++y)
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

/*!
 * \param width        Width of the output image.
 * \param height       Height of the output image.
 * \param imageCount   Number of output images.
 * \param kernelSize   Width and Height of the kernel
 * \param image        The image buffer, (width+kernelSize)*(height+kernelSize)*imageCount*3 entries
 * \param weights      The weight buffer, width*height*kernelSize entries
 * \param out          The output buffer, width*height*imageCount*3 entries
 * */
void WeightedAverageKernelLauncher(
		size_t width, size_t height, size_t imageCount, size_t kernelSize,
		float const * image, float const * weights,
		float * out);

class WeightedAverageGPUOp : public OpKernel {
public:
  explicit WeightedAverageGPUOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input 'images' tensor.
    const Tensor &images_tensor = context->input(0);
    OP_REQUIRES(context, images_tensor.shape().dims() == 4,
                errors::InvalidArgument("The input 'images' is not a 4-D tensor."));
    auto images = images_tensor.tensor<float, 4>();

    // Grab the input 'weights' tensor.
    const Tensor &weights_tensor = context->input(1);
    OP_REQUIRES(context, weights_tensor.shape().dims() == 4,
                errors::InvalidArgument("The input 'weights' is not a 4-D tensor."));
    auto weights = weights_tensor.tensor<float, 4>();

    auto const nbatches    = weights.dimension(0);
    auto const kernel_area = weights.dimension(3);
    auto const kernel_size = static_cast<Eigen::Index>(sqrt(kernel_area));

    // Ensure the 'images' and 'weights' inputs are consistent.
    OP_REQUIRES(context, images.dimension(0) == nbatches,
                errors::InvalidArgument("Images and weights should have the same number of batches."));
    OP_REQUIRES(context, images.dimension(3) == 3,
                errors::InvalidArgument("Images should be RGB (have 3 dimensions)."));

    // Ensure the kernel is a square with an odd-sized side.
    OP_REQUIRES(context, kernel_size % 2 == 1,
                errors::InvalidArgument("Kernel size must be odd."));
    OP_REQUIRES(context, kernel_area == kernel_size * kernel_size,
                errors::InvalidArgument("Kernel area must be square of an odd number."));

    // The image buffer is assumed to be padded with a border equal to the
    // kernel radius.
    auto const border     = (kernel_size - 1)/2;
    auto const img_width  = images.dimension(2);
    auto const img_height = images.dimension(1);
    auto const wgt_width  = img_width  - 2 * border;
    auto const wgt_height = img_height - 2 * border;

    // Ensure the image buffer is properly padded.
    OP_REQUIRES(context, weights.dimension(2) == wgt_width && weights.dimension(1) == wgt_height,
		errors::InvalidArgument("Images and weights should have the same image size minus the chop-off-border."));

    // Create the output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {nbatches, wgt_height, wgt_width, 3}, &output_tensor));
    auto output = output_tensor->tensor<float, 4>();

    // Perform the weighted averaging
	WeightedAverageKernelLauncher(wgt_width, wgt_height, nbatches, kernel_size,
			images.data(), weights.data(), output.data());
  }
};

REGISTER_KERNEL_BUILDER(
  Name("WeightedAverage").Device(DEVICE_GPU),
  WeightedAverageGPUOp
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
    // Grab the input 'weights' tensor.
    const Tensor &weights_tensor = context->input(0);
    OP_REQUIRES(context, weights_tensor.shape().dims() == 4,
                errors::InvalidArgument("The input 'weights' is not a 4-D tensor."));
    auto weights = weights_tensor.tensor<float, 4>();

    // Grab the input 'grads' tensor.
    const Tensor &grads_tensor = context->input(1);
    auto grads = grads_tensor.tensor<float, 4>();

    // Grab the input 'colors' tensor.
    const Tensor &colors_tensor = context->input(2);
    auto colors = colors_tensor.tensor<float, 4>();

    // Create the output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, weights_tensor.shape(), &output_tensor));
    auto output = output_tensor->tensor<float, 4>();

    const int nbatches    = weights.dimension(0);
    const int width       = weights.dimension(1);
    const int height      = weights.dimension(2);
    const int kernel_area = weights.dimension(3);
    const int kernel_size = sqrt(kernel_area);

    #pragma omp parallel for
    for (int b = 0; b < nbatches; ++b)
      for (int x = 0; x < width; ++x)
        for (int y = 0; y < height; ++y)
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
)

/*!
 * \param width        Width of the output image.
 * \param height       Height of the output image.
 * \param imageCount   Number of output images.
 * \param kernelSize   Width and Height of the kernel
 * \param gradients    The gradient buffer
 * \param image        The image buffer
 * \param out          The output buffer
 * */
void WeightedAverageGradientsKernelLauncher(
		size_t width, size_t height, size_t imageCount, size_t kernelSize,
		float const * gradients, float const * image,
		float * out);

class WeightedAverageGradientsGPUOp : public OpKernel {
public:
  explicit WeightedAverageGradientsGPUOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {

	// Grab the input 'weights' tensor.
    const Tensor &weights_tensor = context->input(0);
    OP_REQUIRES(context, weights_tensor.shape().dims() == 4,
                errors::InvalidArgument("The input 'weights' is not a 4-D tensor."));
    auto weights = weights_tensor.tensor<float, 4>();

    // Grab the input 'grads' tensor.
    const Tensor &grads_tensor = context->input(1);
    auto grads = grads_tensor.tensor<float, 4>();

    // Grab the input 'colors' tensor.
    const Tensor &colors_tensor = context->input(2);
    auto colors = colors_tensor.tensor<float, 4>();

    // Create the output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, weights_tensor.shape(), &output_tensor));
    auto output = output_tensor->tensor<float, 4>();

    auto const nbatches = weights.dimension(0);
    auto const height = weights.dimension(1);
    auto const width = weights.dimension(2);
    auto const kernel_area = weights.dimension(3);
    auto const kernel_size = static_cast<Eigen::Index>(sqrt(kernel_area));

   // Ensure the 'colors' and 'weights' inputs are consistent.
    OP_REQUIRES(context, colors.dimension(0) == nbatches,
                errors::InvalidArgument("Colors and weights should have the same number of batches."));
    OP_REQUIRES(context, colors.dimension(3) == 3,
                errors::InvalidArgument("Colors should be RGB (have 3 dimensions)."));

    // Ensure the kernel is the a square with an odd-sized side.
    OP_REQUIRES(context, kernel_size % 2 == 1,
                errors::InvalidArgument("Kernel size must be odd."));
    OP_REQUIRES(context, kernel_area == kernel_size * kernel_size,
                errors::InvalidArgument("Kernel area must be square of an odd number."));

    // Perform the weighted averaging
	WeightedAverageGradientsKernelLauncher(width, height, nbatches, kernel_size,
			grads.data(), colors.data(), output.data());
  }
};

REGISTER_KERNEL_BUILDER(
		Name("WeightedAverageGradients").Device(DEVICE_GPU),
		WeightedAverageGradientsGPUOp
)
