// Computes the raw array index from size and ND index
__device__ constexpr
size_t index(size_t const (&dims)[4], size_t i0, size_t i1, size_t i2, size_t i3)
{
	return dims[3] * (dims[2] * (dims[1] * i0 + i1) + i2) + i3;
}

__device__ constexpr
size_t index(size_t const (&dims)[4], dim3 const & i)
{
	return index(dims, i.z, i.y, i.x, 0);
}

__global__
void WeightedAverageKernel(
		size_t width, size_t height, size_t kernelSize,
		float const * __restrict__ image, float const * __restrict__ weights,
		float * __restrict__ out)
{
	extern __shared__ float imageBuffer[];

	dim3 blockPixelIdx(
			blockIdx.x * blockDim.x,
			blockIdx.y * blockDim.y,
			blockIdx.z * blockDim.z
	);

	dim3 blockSize(
			blockPixelIdx.x + blockDim.x > width ? width - blockPixelIdx.x : blockDim.x,
			blockPixelIdx.y + blockDim.y > height ? height - blockPixelIdx.y : blockDim.y,
			1
	);

	dim3 pixelIdx(
			blockPixelIdx.x + threadIdx.x,
			blockPixelIdx.y + threadIdx.y,
			blockPixelIdx.z + threadIdx.z
	);

	// Load shared data
	// Data is RowMajor: batchsize x height x width x channels
	size_t const depth = 3;
	size_t const imageDims[4] = {
		blockDim.z * gridDim.z,
		height + kernelSize - 1,
		width + kernelSize - 1,
		depth
	};

	size_t const imageStart = index(imageDims, blockPixelIdx);
	size_t const imageStride = imageDims[3] * imageDims[2];
	size_t const bufferWidth = (blockSize.x + kernelSize - 1) * depth; // in floats
	size_t const bufferHeight = (blockSize.y + kernelSize - 1); // in columns

	for (size_t j = 0; j < bufferHeight; j += blockDim.y)
	{
		size_t idxY = j + threadIdx.y;
		if (idxY < bufferHeight)
		{
			for (size_t i = 0; i < bufferWidth; i += blockDim.x)
			{
				size_t idxX = i + threadIdx.x;
				if (idxX < bufferWidth)
					imageBuffer[idxY * bufferWidth + idxX] = image[imageStart + idxY * imageStride + idxX];
			}
		}
	}

	// Compute Result
	__syncthreads();
	
	// Data is RowMajor: nbatches x height x width x kernelarea
	size_t const kernelDims[4] = {
		blockDim.z * gridDim.z,
		height,
		width,
		kernelSize * kernelSize
	};
	size_t const outDims[4] = {
		kernelDims[0],
		kernelDims[1],
		kernelDims[2],
		depth
	};

	// Invalid output pixel, since the input size is not a multiple of the blockDim
	if (pixelIdx.x >= outDims[2] || pixelIdx.y >= outDims[1])
		return;

	size_t const kernelStart = index(kernelDims, pixelIdx);
	size_t const bufferStart = bufferWidth * threadIdx.y + depth * threadIdx.x;

	float outData[depth] = {};
	for (size_t patchy = 0; patchy < kernelSize; ++patchy)
	{
		size_t const patchRow = kernelStart + kernelSize * patchy;
		size_t const bufferRow = bufferStart + patchy * bufferWidth;
		for (size_t patchx = 0; patchx < kernelSize; ++patchx)
		{
			// For some reason, kernels area column major
			float const weight = weights[patchRow + patchx];
			size_t const bufferIdx = bufferRow + patchx * depth;
			outData[0] += weight * imageBuffer[bufferIdx + 0];
			outData[1] += weight * imageBuffer[bufferIdx + 1];
			outData[2] += weight * imageBuffer[bufferIdx + 2];
		}
	}

	// Write output
	size_t outStart = index(outDims, pixelIdx);
	out[outStart + 0] = outData[0];
	out[outStart + 1] = outData[1];
	out[outStart + 2] = outData[2];
}

void WeightedAverageKernelLauncher(
		size_t width, size_t height, size_t imageCount, size_t kernelSize,
		float const * image, float const * weights,
		float * out)
{
	size_t const blockBase = 32;
	dim3 blockDim(blockBase, blockBase, 1);
	// Grid Dimension rounds up to neirest full block size
	dim3 gridDim((width + blockBase - 1) / blockDim.x, (height + blockBase - 1) / blockDim.y, imageCount);
	size_t sharedSize = (blockDim.x + kernelSize - 1) * (blockDim.y + kernelSize - 1) * 3 * sizeof(float);
	WeightedAverageKernel<<<gridDim, blockDim, sharedSize>>>(width, height, kernelSize, image, weights, out);
}

__global__
void WeightedAverageGradientsKernel(
		size_t width, size_t height, size_t kernelSize,
		float const * __restrict__ gradients, float const * __restrict__ image,
		float * __restrict__ out)
{
	extern __shared__ float imageBuffer[];

	dim3 blockPixelIdx(
			blockIdx.x * blockDim.x,
			blockIdx.y * blockDim.y,
			blockIdx.z * blockDim.z
	);

	dim3 blockSize(
			blockPixelIdx.x + blockDim.x > width ? width - blockPixelIdx.x : blockDim.x,
			blockPixelIdx.y + blockDim.y > height ? height - blockPixelIdx.y : blockDim.y,
			1
	);

	dim3 pixelIdx(
			blockPixelIdx.x + threadIdx.x,
			blockPixelIdx.y + threadIdx.y,
			blockPixelIdx.z + threadIdx.z
	);

	// Load shared data
	// Data is RowMajor: batchsize x height x width x channels
	size_t const depth = 3;
	size_t const imageDims[4] = {
		blockDim.z * gridDim.z,
		height + kernelSize - 1,
		width + kernelSize - 1,
		depth
	};

	size_t const imageStart = index(imageDims, blockPixelIdx);
	size_t const imageStride = imageDims[3] * imageDims[2];
	size_t const bufferWidth = (blockSize.x + kernelSize - 1) * depth; // in floats
	size_t const bufferHeight = (blockSize.y + kernelSize - 1); // in columns

	for (size_t j = 0; j < bufferHeight; j += blockDim.y)
	{
		size_t idxY = j + threadIdx.y;
		if (idxY < bufferHeight)
		{
			for (size_t i = 0; i < bufferWidth; i += blockDim.x)
			{
				size_t idxX = i + threadIdx.x;
				if (idxX < bufferWidth)
					imageBuffer[idxY * bufferWidth + idxX] = image[imageStart + idxY * imageStride + idxX];
			}
		}
	}

	// Compute Result
	__syncthreads();

	// Data is RowMajor: nbatches x height x width x depth
	size_t const gradDims[4] = {
		blockDim.z * gridDim.z,
		height,
		width,
		depth
	};
	size_t const outDims[4] = {
		gradDims[0],
		gradDims[1],
		gradDims[2],
		kernelSize*kernelSize
	};

	// Invalid output pixel, since the input size is not a multiple of the blockDim
	if (pixelIdx.x >= outDims[2] || pixelIdx.y >= outDims[1])
		return;

	size_t const gradStart = index(gradDims, pixelIdx);
	size_t const bufferStart = bufferWidth * threadIdx.y + depth * threadIdx.x;
	float gradient[depth] = {
		gradients[gradStart + 0],
		gradients[gradStart + 1],
		gradients[gradStart + 2]
	};
	size_t outStart = index(outDims, pixelIdx);
	for (size_t patchy = 0; patchy < kernelSize; ++patchy)
	{
		size_t const bufferRow = bufferStart + patchy * bufferWidth;
		size_t const outRow = outStart + patchy * kernelSize;
		for (size_t patchx = 0; patchx < kernelSize; ++patchx)
		{
			size_t const bufferIdx = bufferRow + patchx * depth;
			size_t const outIdx = outRow + patchx;
			float result = 0;
			for (size_t i = 0; i < depth; ++i)
			{
				result += gradient[i] * imageBuffer[bufferIdx + i];
			}
			out[outIdx] = result;
		}
	}
}

void WeightedAverageGradientsKernelLauncher(
		size_t width, size_t height, size_t imageCount, size_t kernelSize,
		float const * gradients, float const * image,
		float * out)
{
	size_t const blockBase = 32;
	dim3 blockDim(blockBase, blockBase, 1);
	// Grid Dimension rounds up to neirest full block size
	dim3 gridDim((width + blockBase - 1) / blockDim.x, (height + blockBase - 1) / blockDim.y, imageCount);
	size_t sharedSize = (blockDim.x + kernelSize - 1) * (blockDim.y + kernelSize - 1) * 3 * sizeof(float);
	WeightedAverageGradientsKernel<<<gridDim, blockDim, sharedSize>>>(width, height, kernelSize, gradients, image, out);
}
