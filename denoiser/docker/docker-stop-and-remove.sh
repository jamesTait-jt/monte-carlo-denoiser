read -r -p "Did you remember to push your changes? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    sudo docker stop Tensorflow
    sudo docker rm Tensorflow
else
    sudo docker exec -it Tensorflow bash
fi

