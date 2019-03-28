# Tungsten with diffuse/specular separation

This is a modified version of the open source [Tungsten](https://github.com/tunabrain/tungsten) renderer. It adds support for additional output buffers 'diffuse' and 'specular' which add up to the final color. We also ship a utility script, `tungsten.py`, that packages Tungsten output in the right format for use with the published code for KPCN.

Note: `tungsten.py` overrules some settings of the `scene.json` files to make sure that its outputs are in the right format. Some settings, like resolution or spp have to be set through the script, rather than the config file.

## Use with Docker (recommended)

Docker provides a lightweight alternative to virtualization. It creates containers that run in an isolated environment. Such containers have access to the host file system through explicit 'mounts'. We provide a Dockerfile which can be used to create a Docker image. At every use, a fresh container is made from this image. Tungsten runs inside this container, and the container is automatically deleted after the render finishes.

### Installation

1. Install docker according to the instructions [here](https://docs.docker.com/engine/installation/).
2. In the root of this directory, create a Docker image and name it 'tungsten': `sudo docker build -t tungsten .` This step copies all source files to the image. If you change any files, rebuild the image with the same command.
3. Run the script in a docker container. First, download a scene from [Benedikt Bitterli's website](https://benedikt-bitterli.me/resources/), and store it on the host filesystem at for example `/data/testscene`. Now run

```bash
sudo docker run -v /data/testscene:/scene --rm -it tungsten \
scene.json out.exr --threads 10 --spp 32
```

Here `--rm` makes sure the container is deleted after rendering, `-v` specifies that the scene directory is mounted in the container under `/scene`. The `out.exr` file will be available to the host filesystem in `/data/testscene/out.exr`. If you would like to write the output somewhere else, you can mount additional directories:

```bash
sudo docker run -v /data/testscene:/scene -v /data/custom/output:/out --rm -it tungsten \
scene.json /out/out.exr --threads 10 --spp 32 --seed 234
```

To see all options of the `tungsten.py` script, run

```bash
sudo docker run -v /data/testscene:/scene --rm -it tungsten --help
```

## Without Docker

Follow the instructions inside `Dockerfile` to install all dependencies and build Tungsten. Set the environment variable `TUNGSTEN_BINARY` and call the tungsten.py script like this:

```bash
tungsten.py scene.json out.exr --threads 10 --spp 32
```

## Output channels

This script generates multi-channel EXR files with the following channels:

 - albedo                        : albedo.R, albedo.G, albedo.B,
 - albedoA                       : albedoA.R, albedoA.G, albedoA.B,
 - albedoVariance                : albedoVariance.Z,
 - colorA                        : colorA.R, colorA.G, colorA.B,
 - colorVariance                 : colorVariance.Z,
 - default                       : R, G, B,
 - depth                         : depth.Z,
 - depthA                        : depthA.Z,
 - depthVariance                 : depthVariance.Z,
 - diffuse                       : diffuse.R, diffuse.G, diffuse.B,
 - diffuseA                      : diffuseA.R, diffuseA.G, diffuseA.B,
 - diffuseVariance               : diffuseVariance.Z,
 - normal                        : normal.R, normal.G, normal.B,
 - normalA                       : normalA.R, normalA.G, normalA.B,
 - normalVariance                : normalVariance.Z,
 - specular                      : specular.R, specular.G, specular.B,
 - specularA                     : specularA.R, specularA.G, specularA.B,
 - specularVariance              : specularVariance.Z,
 - visibility                    : visibility.Z,
 - visibilityA                   : visibilityA.Z,
 - visibilityVariance            : visibilityVariance.Z,

`*A` are half buffers: they contain averages of 50% of the samples. `*Variance` are estimates of the variance of the average sample mean.
