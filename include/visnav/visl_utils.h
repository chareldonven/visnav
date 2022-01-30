#pragma once

#include <set>
#include <map>
#include <visnav/common_types.h>

#include <visnav/keypoints.h>
#include <visnav/serialization.h>

namespace visnav {

// update visualisation. If the track is longer then tracklength, old points are
// removed.
void updateVisualisationTracks(const TrackedPoints& trackedPoints,
                               const size_t tracklength,
                               const KeypointsPositions& kd,
                               VisualisationTracks& visualisationTracks) {
  VisualisationTracks newVisualisationTracks;

  for (const auto& trackedPoint : trackedPoints) {
    // get all pointstrack of the current point
    PointsOfTrack pointsOfTrack = visualisationTracks[trackedPoint.second];
    // insert newest point
    pointsOfTrack.insert(pointsOfTrack.begin(), kd[trackedPoint.first]);
    // remove old point
    if (pointsOfTrack.size() > tracklength) pointsOfTrack.pop_back();
    newVisualisationTracks[trackedPoint.second] = pointsOfTrack;
  }
  visualisationTracks = newVisualisationTracks;
}

double points2Angle(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
  double pi = 3.14159265;
  Eigen::Vector2d p0;
  p0[0] = p1[0];
  p0[1] = p1[1] + 1;

  Eigen::Vector2d w = p2 - p1;
  Eigen::Vector2d v;
  v[0] = 0;
  v[1] = w.norm();

  double dot = w[0] * v[0] + w[1] * v[1];
  double det = w[1] * v[0] - w[0] * v[1];
  double angle = atan2(det, dot) * 180.0 / pi + 180.0;
  return angle;
}

void angle2rgb(double h, int& r, int& g, int& b) {
  double x = 1 - abs(fmod(h / 60.0, 2.0) - 1);

  if (h < 60) {
    r = 255;
    g = (int)(x * 255);
    b = 0;

  } else if (h < 120) {
    r = (int)(x * 255);
    g = 255;
    b = 0;
  } else if (h < 180) {
    r = 0;
    g = 255;
    b = (int)(x * 255);
  } else if (h < 240) {
    r = 0;
    g = (int)(x * 255);
    b = 255;
  } else if (h < 300) {
    r = (int)(x * 255);
    g = 0;
    b = 255;
  } else {
    r = 255;
    g = 0;
    b = (int)(255 * x);
  }
}

}  // namespace visnav
