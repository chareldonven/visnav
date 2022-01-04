/*

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  const string about =
      "This sample demonstrates Lucas-Kanade Optical Flow calculation.\n"
      "The example file can be downloaded from:\n"
      "  "
      "https://www.bogotobogo.com/python/OpenCV_Python/images/"
      "mean_shift_tracking/slow_traffic_small.mp4";
  const string keys =
      "{ h help |      | print this help message }"
      "{ @image | vtest.avi | path to image file }";
  CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  string filename = "slow_traffic_small.mp4";
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  VideoCapture capture(filename);
  if (!capture.isOpened()) {
    // error in opening the video input
    cerr << "Unable to open file!" << endl;
    return 0;
  }

  // Create some random colors
  vector<Scalar> colors;
  RNG rng;
  for (int i = 0; i < 100; i++) {
    int r = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int b = rng.uniform(0, 256);
    colors.push_back(Scalar(r, g, b));
  }

  Mat old_frame, old_gray;
  vector<Point2f> p0, p1;

  // Take first frame and find corners in it
  capture >> old_frame;
  cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
  goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

  // Create a mask image for drawing purposes
  Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

  while (true) {
    Mat frame, frame_gray;

    capture >> frame;
    if (frame.empty()) break;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria =
        TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err,
                         Size(15, 15), 2, criteria);

    vector<Point2f> good_new;
    for (uint i = 0; i < p0.size(); i++) {
      // Select good points
      if (status[i] == 1) {
        good_new.push_back(p1[i]);
        // draw the tracks
        line(mask, p1[i], p0[i], colors[i], 2);
        circle(frame, p1[i], 5, colors[i], -1);
      }
    }
    Mat img;
    add(frame, mask, img);

    imshow("Frame", img);

    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) break;

    // Now update the previous frame and previous points
    old_gray = frame_gray.clone();
    p0 = good_new;
  }
} */

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
using namespace cv;
using namespace std;
static void help() {
  // print a welcome message, and the OpenCV version
  cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
          "Using OpenCV version "
       << CV_VERSION << endl;
  cout << "\nIt uses camera by default, but you can provide a path to video as "
          "an argument.\n ";
  cout << "\nHot keys : \n"
          "\tESC - quit the program\n"
          "\tr - auto-initialize tracking\n"
          "\tc - delete all the points\n"
          "\tn - switch the \"night\" mode on/off\n"
          "To add/remove a feature point click it\n"
       << endl;
}
Point2f point;
bool addRemovePt = false;
static void onMouse(int event, int x, int y, int, void*)

{
  if (event == EVENT_LBUTTONDOWN) {
    point = Point2f((float)x, (float)y);
    addRemovePt = true;
  }
}
int main(int argc, char** argv) {
  VideoCapture cap;
  TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
  Size subPixWinSize(10, 10), winSize(31, 31);
  const int MAX_COUNT = 500;
  bool needToInit = false;
  bool nightMode = false;
  help();

  cv::CommandLineParser parser(argc, argv, "{@input|0|}");
  string input = parser.get<string>("@input");
  if (input.size() == 1 && isdigit(input[0]))
    cap.open(input[0] - '0');
  else
    cap.open(input);
  if (!cap.isOpened()) {
    cout << "Could not initialize capturing...\n";
    return 0;
  }
  namedWindow("LK Demo", 1);
  setMouseCallback("LK Demo", onMouse, 0);
  Mat gray, prevGray, image, frame;
  vector<Point2f> points[2];
  for (;;) {
    cap >> frame;
    if (frame.empty()) break;
    frame.copyTo(image);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    if (nightMode) image = Scalar::all(0);
    if (needToInit) {
      // automatic initialization
      goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0,
                          0.04);
      cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
      addRemovePt = false;
    } else if (!points[0].empty()) {
      vector<uchar> status;
      vector<float> err;
      if (prevGray.empty()) gray.copyTo(prevGray);
      calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err,
                           winSize, 3, termcrit, 0, 0.001);
      size_t i, k;
      for (i = k = 0; i < points[1].size(); i++) {
        if (addRemovePt) {
          if (norm(point - points[1][i]) <= 5) {
            addRemovePt = false;
            continue;
          }
        }
        if (!status[i]) continue;
        points[1][k++] = points[1][i];
        circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
      }
      points[1].resize(k);
    }
    if (addRemovePt && points[1].size() < (size_t)MAX_COUNT) {
      vector<Point2f> tmp;
      tmp.push_back(point);
      cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
      points[1].push_back(tmp[0]);
      addRemovePt = false;
    }
    needToInit = false;
    imshow("LK Demo", image);
    char c = (char)waitKey(10);
    if (c == 27) break;
    switch (c) {
      case 'r':
        needToInit = true;
        break;
      case 'c':
        points[0].clear();
        points[1].clear();
        break;
      case 'n':
        nightMode = !nightMode;
        break;
    }
    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);
  }
  return 0;
}
