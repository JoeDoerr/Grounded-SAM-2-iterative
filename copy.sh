dockid=$1
docker cp ./mydemo.py $dockid:/home/appuser/Grounded-SAM-2/mydemo.py
docker cp ./server_gsam.py $dockid:/home/appuser/Grounded-SAM-2/server_gsam.py
docker cp ./test_client_gsam.py $dockid:/home/appuser/Grounded-SAM-2/test_client_gsam.py
docker cp ./sam2/sam2_video_predictor.py $dockid:/home/appuser/Grounded-SAM-2/sam2/sam2_video_predictor.py
docker cp ./sam2/utils/misc.py $dockid:/home/appuser/Grounded-SAM-2/sam2/utils/misc.py
docker cp ./00001.jpg $dockid:/home/appuser/Grounded-SAM-2/
docker cp ./00002.jpg $dockid:/home/appuser/Grounded-SAM-2/
docker cp ./00003.jpg $dockid:/home/appuser/Grounded-SAM-2/
docker cp $dockid:/home/appuser/Grounded-SAM-2/output00001.jpg .
docker cp $dockid:/home/appuser/Grounded-SAM-2/output00002.jpg .
docker cp $dockid:/home/appuser/Grounded-SAM-2/output00003.jpg .
