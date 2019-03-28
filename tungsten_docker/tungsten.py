#!/usr/bin/env python

import json
import tempfile
import subprocess
import shutil
import os, sys
from time import time
import pyexr
import random
import click

TUNGSTEN = os.getenv('TUNGSTEN_BINARY', "tungsten")

@click.command()
@click.argument("scene", type=click.Path(exists=True))
@click.argument("out", type=click.Path(exists=False))
@click.option("--datadir", default=None, type=str, help="Directory root for loaded assets")
@click.option("--with-feature-buffers/--without-feature-buffers", default=True, help="Store normal, albedo, visibility and depth buffers")
@click.option("--with-half-buffers/--without-half-buffers", default=True, help="Store a buffer with half the samples, as a variance estimate.")
@click.option("--with-variance/--without-variance", default=True, help="Store a per-pixel variance of the samples.")
@click.option("--spp", default=256, type=int)
@click.option("--resolution", default=[1280,720], type=int, nargs=2)
@click.option("--min-bounces", default=0, type=int)
@click.option("--max-bounces", default=64, type=int)
@click.option("--spp-step", default=16, type=int)
@click.option("--resume-file", type=str)
@click.option("--threads", default=1, type=int)
@click.option("--seed", default=None, type=int)
def cli(*args, **kwargs):
  render(*args, **kwargs)

def render(scene,
           out,
           datadir=None,
           with_feature_buffers=True,
           with_half_buffers=True,
           with_variance=True,
           spp=256,
           resolution=[1280, 720],
           max_bounces=64,
           min_bounces=0,
           spp_step=16,
           threads=1,
           seed=None,
           resume_file=None):

    if datadir is None:
        datadir = os.path.realpath(os.path.dirname(scene))
    else:
        datadir = os.path.realpath(datadir)

    if seed is None:
        seed = random.randint(0, 1000000000000)

    buffers = ['color']
    extra_buffers = ['normal','albedo','visibility','depth','diffuse','specular']
    if with_feature_buffers:
        buffers = buffers + extra_buffers

    output_buffers = [
        {"type": s,
         "hdr_output_file": "%s.exr"%s,
         "sample_variance": with_variance,
         "two_buffer_variance": with_half_buffers
        } for s in buffers
    ]

    scene = json.load(open(scene, 'r'))

    _make_files_absolute(scene, datadir)

    scene["integrator"] = {
        "type": "path_tracer",
        "min_bounces": min_bounces,
        "max_bounces": max_bounces,
        "enable_consistency_checks": True,
        "enable_two_sided_shading": True,
        "enable_light_sampling": True,
        "enable_volume_light_sampling": True
    }
    scene["renderer"] = {
        "overwrite_output_files": True,
        "adaptive_sampling": False,
        "stratified_sampler": False,
        "scene_bvh": True,
        "checkpoint_interval": "10m",
        "spp": spp,
        "spp_step": spp_step,
        "output_buffers": output_buffers
    }

    if resume_file:
        resume_file = os.path.abspath(resume_file)
        scene["renderer"]["enable_resume_render"] = True
        scene["renderer"]["resume_render_file"] = resume_file

    scene['camera']['resolution'] = resolution

    # Write the scene file to a temporary directory,
    # run Tungsten on it
    # Combine al buffers into one EXR
    # and write it to the desired location
    try:
        tmp = tempfile.mkdtemp()
        scene_file = os.path.join(tmp, "scene.json")
        json.dump(scene, open(scene_file,'w'), indent=1)
        subprocess.check_call([TUNGSTEN, scene_file,
                               '--output-directory', tmp,
                               '--threads', str(threads),
                               '--seed', str(seed)], cwd=tmp)
        print("Combining EXR output")
        start = time()
        data = {'default': pyexr.read(os.path.join(tmp, 'color.exr'))}
        if with_variance:
            data['colorVariance'] = pyexr.read(os.path.join(tmp, 'colorVariance.exr'))
        if with_half_buffers:
            data['colorA'] = pyexr.read(os.path.join(tmp, 'colorA.exr'))
        if with_feature_buffers:
            for b in extra_buffers:
                data[b] = pyexr.read(os.path.join(tmp, '%s.exr' % b))
                if with_variance:
                    data['%sVariance'%b] = pyexr.read(os.path.join(tmp, '%sVariance.exr'%b))
                if with_half_buffers:
                    data['%sA'%b] = pyexr.read(os.path.join(tmp, '%sA.exr'%b))

        base, ext = os.path.splitext(out)
        counter = 0
        while os.path.exists(out):
          counter += 1
          out = "%s.%03d%s" % (base, counter, ext)
        out = os.path.abspath(out)
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        print("Writing to '%s'" % out)
        pyexr.write(out, data)

        print("Done. Took %d s." % (time() - start))

        if resume_file and os.path.isfile(resume_file):
            os.remove(resume_file)
    finally:
        shutil.rmtree(tmp)


def _make_files_absolute(root, basepath):
  def makeabs(data, keys=[]):
    if isinstance(data, dict):
      for i, val in data.items():
        makeabs(val, keys=keys+[i])
    elif isinstance(data, list):
      for i, val in enumerate(data):
        makeabs(val, keys=keys+[i])
    elif isinstance(data, basestring) and '/' in data:
      x = root
      for key in keys[:-1]:
        x = x[key]
      if isinstance(x, dict):
        x[keys[-1]] = os.path.join(basepath, data)

  return makeabs(root)

if __name__ == "__main__":
    cli()
