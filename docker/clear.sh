#!/bin/bash
docker ps | awk '{ print $1,$2 }' | grep max/pomrl | awk '{print $1 }' | xargs -I {} docker stop {}
docker ps -a | awk '{ print $1,$2 }' | grep max/pomrl | awk '{print $1 }' | xargs -I {} docker rm {}
