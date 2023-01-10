import cv2, time
# import pafy
from ultrafastLaneDetector.utils import ModelType
from ultrafastLaneDetector.ultrafastLane import UltrafastLaneDetector
from ultrafastLaneDetector.ultrafastLaneV2 import UltrafastLaneDetectorV2


video_path = "./temp/test.mp4"
model_path = "models/culane_res34.onnx"
# model_type = ModelType.UFLD_TUSIMPLE
model_type = ModelType.UFLDV2_CULANE

if __name__ == "__main__":
	# Initialize video
	cap = cv2.VideoCapture(video_path)
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Initialize lane detection model
	print("Model Type : ", model_type.name)
	if ( "UFLDV2" in model_type.name) :
		lane_detector = UltrafastLaneDetectorV2(model_path, model_type)
	else :
		lane_detector = UltrafastLaneDetector(model_path, model_type)

	cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (width, height))
	fps = 0
	frame_count = 0
	start = time.time()
	while cap.isOpened():
		try:
			# Read frame from the video
			ret, frame = cap.read()
		except:
			continue

		if ret:	

			# Detect the lanes
			output_img = lane_detector.AutoDrawLanes(frame)

			frame_count += 1
			if frame_count >= 30:
				end = time.time()
				fps = frame_count / (end - start)
				frame_count = 0
				start = time.time()
			cv2.putText(output_img, "FPS: %.2f" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow("Detected lanes", output_img)

		else:
			break
		vout.write(output_img)	
		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	vout.release()
	cap.release()
	cv2.destroyAllWindows()