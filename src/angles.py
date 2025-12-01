from pathlib import Path
import numpy as np
import math
import os
import re
from vision.py import angle_at_shoulder_deg

### this is the script used to read the output .txt
### file into usable angles

ROOT = Path(__file__).resolve().parent
coords_path = os.path.join(ROOT,'data/coords.txt')

def create_landmark_dict():
	mark_dict = {
			"left shoulder" : [],
			"left wrist" : [],
			"left hip" : [],
			"right shoulder" : [],
			"right wrist" : [],
			"right hip" : []
	}
	#to index into landmark map w/ ints)
	global int_map
	int_map = {index: value for index, value in enumerate(mark_dict.keys())}
	
	value_pattern = r"(-?\d.\d{3,10})"
	with open(coords_path, "r") as f:
		raw = f.readlines()
		lines = raw[0::5]
		idx = 0
		dict_i = 0
		for line in lines:
			if(dict_i == 6): #reset iteration through dict map
				dict_i = 0
			
			coords = []
			for jdx in range(1,4): #gather set of 3 coordinates
				val = re.findall(value_pattern, raw[idx+jdx])
				if val:
					coords.append(float(val[0]))
			mark_dict[int_map[dict_i]].append(coords)
			
			idx += 5
			dict_i += 1
	return mark_dict

if __name__ == "__main__":
	coords_dict = create_landmark_dict()

	for idx in range(0,len(coords_dict['left shoulder'])):
		ls = coords_dict['left shoulder'][idx]
		lw = coords_dict['left wrist'][idx]
		lh = coords_dict['left hip'][idx]
		
		rs = coords_dict['right shoulder'][idx]
		rw = coords_dict['right wrist'][idx]
		rh = coords_dict['right hip'][idx]

		#rangle = angle_at_shoulder_deg(rs,rh,rw)
		#print(f"right angle: {rangle}")
		langle = angle_at_shoulder_deg(ls,lh,lw)
		print(f"left angle: {langle}")
