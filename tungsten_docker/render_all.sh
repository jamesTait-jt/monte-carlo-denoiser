### ------ TRAIN ------ ###

# Living room 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/living-room:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
 #   scene.json /out/living-room_32.exr --threads 4 --spp 32

# Living room 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/living-room:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/living-room_1024exr --threads 4 --spp 1024

# Car2 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/car2:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/car2_32.exr --threads 4 --spp 32

# Car2 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/car2:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/car2_1024exr --threads 4 --spp 1024

# Classroom 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/classroom:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/classroom_32.exr --threads 4 --spp 32

# Classroom 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/classroom:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/classroom_1024exr --threads 4 --spp 1024

# House 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/house:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/house_32.exr --threads 4 --spp 32

# House 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/house:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/house_1024exr --threads 4 --spp 1024

# Spaceship 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/spaceship:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
 #   scene.json /out/spaceship_32.exr --threads 4 --spp 32

# Spaceship 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/spaceship:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/spaceship_1024exr --threads 4 --spp 1024

# Staircase 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/staircase:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/staircase_32.exr --threads 4 --spp 32

# Staircase 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/staircase:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/staircase_1024exr --threads 4 --spp 1024

# Bathroom 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/bathroom:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
 #   scene.json /out/bathroom_32.exr --threads 4 --spp 32

# Bathroom 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/bathroom:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/bathroom_1024exr --threads 4 --spp 1024

# Dining room 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/dining-room:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/dining-room_32.exr --threads 4 --spp 32

# Dining room 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/train/dining-room:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/dining-room_1024exr --threads 4 --spp 1024


### ------ TEST ------ ###

# Bathroom2 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/bathroom2:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/bathroom2_32.exr --threads 4 --spp 32

# Bathroom2 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/bathroom2:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/bathroom2_1024exr --threads 4 --spp 1024

# Bedroom 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/bedroom:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/bedroom_32.exr --threads 4 --spp 32

# Bedroom 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/bedroom:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/bedroom_1024exr --threads 4 --spp 1024

# Living-room-2 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/living-room-2:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/living-room-2_32.exr --threads 4 --spp 32

# Living-room-2 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/living-room-2:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/living-room-2_1024exr --threads 4 --spp 1024

# Living-room-3 32 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/living-room-3:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/living-room-3_32.exr --threads 4 --spp 32

# Living-room-3 1024 spp
#sudo docker run -v /home/james/git-repos/monte-carlo-raytracer/tungsten_docker/data/test/living-room-3:/scene -v /home/james/git-repos/monte-carlo-raytracer/data/scenes:/out --rm -it tungsten \
#    scene.json /out/living-room-3_1024exr --threads 4 --spp 1024
