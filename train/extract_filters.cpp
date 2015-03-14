#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "../utils.h"

using namespace cv;
using namespace std;

/*This is part of the implementation of the paper "Text Detection and Character Recognition with
  Unsupervised Feature Learning" by A. Coates et al. in ICDAR2011*/


// Train first layer filters with kmeans
int main (int argc, char* argv[])
{

  if (argc < 2)
  {
     cout << argv[0] << ": Train first layer filters with kmeans." << endl;
     cout << "   Usage " << argv[0] << " <dataset file(s) with train images and labels>" << endl;
     cout << "   e.g.  " << argv[0] << " data/characters/icdar/img_ICDAR_train_labels.txt data/characters/chars74k/img_labels.txt data/characters/synthetic/img_labels.txt" << endl;
     exit(0);
  }

  vector<int> labels;
  vector<string> filenames;
  for (int f=1; f<argc; f++)
  {

    ifstream infile(argv[f]);
    string path = string(argv[f]);
    path.erase(path.end()-11,path.end());
    path.append("/");

    int label;
    string filename;
    while (infile >> filename >> label)
    {
      labels.push_back(label);
      filenames.push_back(path+filename);
    }
  }

  int num_images = labels.size();

  cout << "We have " << num_images << " images in dataset. Starting patch extraction and pre-processing ... " << endl;

  int image_width  = 32;
  int image_height = 32;
  int patch_width  = 8;
  int patch_height = 8;
  int num_patches_x_image = 8;
  int num_patches = num_images * num_patches_x_image;
  int n_iter = 500; //maximum number of kmeans iterations to run

  int K = 150; //number of filters (for visualization must be multiple of 16)

  //load training examples and extract patches

  Mat patches = Mat::zeros(num_patches, patch_width*patch_height, CV_64FC1);
  int patch_counter = 0;


  for (int f=0; f<num_images; f++)
  {
    Mat img = imread(filenames[f]);
    if(img.channels() != 3)
      continue;
    cvtColor(img,img,COLOR_RGB2GRAY);
    resize(img,img,Size(image_width,image_height));

    int cnt = 0;
    while (cnt < num_patches_x_image)
    {
      int x = rand() % (img.cols-patch_width);
      int y = rand() % (img.rows-patch_height);
      Mat patch;
      img(Rect(x,y,patch_width,patch_height)).copyTo(patch);
      patch.convertTo(patch, CV_64FC1);

      patch = patch.reshape(0,1);
      patch.copyTo(patches.row(patch_counter));

      patch_counter++;
      cnt++;
    }
  }

  Mat M, P;
  normalizeAndZCA(patches, M, P);


  //Uses dot-product Kmeans to learn a specified number of bases
  Mat centroids;
  run_projection_kmeans(patches, centroids, K, n_iter);

  //discard filters with low variance
  double varthresh = 0.025;
  selectCentroids(centroids, varthresh);
 


  FileStorage fs("first_layer_centroids.xml", FileStorage::WRITE);
  fs << "D" << centroids;
  fs << "M" << M;
  fs << "P" << P;
  fs.release();


  /*Visualize the filter bank*/
  visualizeNatwork(centroids);

}
