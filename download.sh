if [ ! -d mpii_dataset ]; then
    mkdir mpii_dataset
fi
cd mpii_dataset || exit

wget http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz
tar zxvf mpii_human_pose_v1_u12_1.tar.gz
rm -rf mpii_human_pose_v1_u12_1.tar.gz
mv mpii_human_pose_v1_u12_1 mpii_annotations

# Get Images
wget http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
tar zxvf mpii_human_pose_v1.tar.gz
rm -rf mpii_human_pose_v1.tar.gz
mv mpii_human_pose_v1 mpii_images

cd ..
python DatasetDownloader.py