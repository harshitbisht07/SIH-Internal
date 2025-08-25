"""SAMVAAD : Smart Automated Machine for Voice , Accessiblity and Diversity

"""

import os
import cv2
import pickle
import time
import numpy as np
from collections import deque, Counter
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier

