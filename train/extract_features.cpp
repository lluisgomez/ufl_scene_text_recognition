#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "../utils.h"

using namespace cv;
using namespace std;


/*This is part of the implementation of the paper "Text Detection and Character Recognition with
  Unsupervised Feature Learning" by A. Coates et al. in ICDAR2011*/

//Extract features for a set of images. The features are calculated by convolution 
// with precomputed filters and average pooling
int main (int argc, char* argv[])
{

  if (argc < 3)
  {
     cout << argv[0] << ": Extract features for a set of images. " << endl;
     cout << "   Usage " << argv[0] << " <file storage with precomputed filters> <dataset file with train images and labels>" << endl;
     cout << "   e.g.  " << argv[0] << " first_layer_centroids.xml data/img_all_labels.txt" << endl;
     exit(0);
  }

  //Load filters bank and withenning params
  Mat filters, M, P;
  FileStorage fs(argv[1], FileStorage::READ);
  fs["D"] >> filters;
  fs["M"] >> M;
  fs["P"] >> P;
  fs.release();

  int image_size  = 32;
  int quad_size   = 12;
  int patch_size  = 8;
  int num_quads   = 25; //extract 25 quads (12x12) from each image
  int num_tiles   = 25; //extract 25 patches (8x8) from each quad 

  double alpha    = 0.5; //used for feature representation: scalar non-linear function z = max(0, |D*a| - alpha)

  ifstream infile(argv[2]);
  string path = string(argv[2]);
  path.erase(path.end()-11,path.end());
  path.append("/");
  cout << path << endl;

  int label;
  string filename;
  vector<int> labels;
  vector<string> filenames;
  while (infile >> filename >> label)
  {
    labels.push_back(label);
    filenames.push_back(filename);
  }

  int num_images = labels.size();

  //load training examples and extract features
  cout << "We have " << num_images << " images in dataset. Starting patch extraction and pre-processing ... " << endl;


  Mat quad;
  Mat tmp;

  for (int f=0; f<num_images; f++)
  {
    int patch_count = 0;
    vector< vector<double> > data_pool(9); 
    Mat img  = imread(path+filenames[f]);
    if(img.channels() != 3)
      continue;
    cvtColor(img,img,COLOR_RGB2GRAY);
    resize(img,img,Size(image_size,image_size));

    int quad_id = 1;
    for (int q_x=0; q_x<=image_size-quad_size; q_x=q_x+(quad_size/2-1))
    {
      for (int q_y=0; q_y<=image_size-quad_size; q_y=q_y+(quad_size/2-1))
      {
        Rect quad_rect = Rect(q_x,q_y,quad_size,quad_size); 
        img(quad_rect).copyTo(quad);

        //start sliding window (8x8) in each tile and store the patch as row in data_pool
        for (int w_x=0; w_x<=quad_size-patch_size; w_x++)
        {
          for (int w_y=0; w_y<=quad_size-patch_size; w_y++)
          {
            quad(Rect(w_x,w_y,patch_size,patch_size)).copyTo(tmp);
            tmp = tmp.reshape(0,1);
            tmp.convertTo(tmp, CV_64F);
            normalizeAndZCA(tmp,M,P);
            vector<double> patch;
            tmp.copyTo(patch);
            if ((quad_id == 1)||(quad_id == 2)||(quad_id == 6)||(quad_id == 7))
              data_pool[0].insert(data_pool[0].end(),patch.begin(),patch.end());
            if ((quad_id == 2)||(quad_id == 7)||(quad_id == 3)||(quad_id == 8)||(quad_id == 4)||(quad_id == 9))
              data_pool[1].insert(data_pool[1].end(),patch.begin(),patch.end());
            if ((quad_id == 4)||(quad_id == 9)||(quad_id == 5)||(quad_id == 10))
              data_pool[2].insert(data_pool[2].end(),patch.begin(),patch.end());
            if ((quad_id == 6)||(quad_id == 11)||(quad_id == 16)||(quad_id == 7)||(quad_id == 12)||(quad_id == 17))
              data_pool[3].insert(data_pool[3].end(),patch.begin(),patch.end());
            if ((quad_id == 7)||(quad_id == 12)||(quad_id == 17)||(quad_id == 8)||(quad_id == 13)||(quad_id == 18)||(quad_id == 9)||(quad_id == 14)||(quad_id == 19))
              data_pool[4].insert(data_pool[4].end(),patch.begin(),patch.end());
            if ((quad_id == 9)||(quad_id == 14)||(quad_id == 19)||(quad_id == 10)||(quad_id == 15)||(quad_id == 20))
              data_pool[5].insert(data_pool[5].end(),patch.begin(),patch.end());
            if ((quad_id == 16)||(quad_id == 21)||(quad_id == 17)||(quad_id == 22))
              data_pool[6].insert(data_pool[6].end(),patch.begin(),patch.end());
            if ((quad_id == 17)||(quad_id == 22)||(quad_id == 18)||(quad_id == 23)||(quad_id == 19)||(quad_id == 24))
              data_pool[7].insert(data_pool[7].end(),patch.begin(),patch.end());
            if ((quad_id == 19)||(quad_id == 24)||(quad_id == 20)||(quad_id == 25))
              data_pool[8].insert(data_pool[8].end(),patch.begin(),patch.end());
            patch_count++;
          }
        }

        quad_id++;
      }
    }

    //do dot product of each normalized and whitened patch 
    //each pool is averaged and this yields a representation of 9xD 
    Mat feature = Mat::zeros(9,filters.rows,CV_64FC1);
    for (int i=0; i<9; i++)
    {
      Mat pool = Mat(data_pool[i]);
      pool = pool.reshape(0,data_pool[i].size()/filters.cols);
      for (int p=0; p<pool.rows; p++)
      {
        for (int f=0; f<filters.rows; f++)
        {
          feature.row(i).at<double>(0,f) = feature.row(i).at<double>(0,f) + max(0.0,std::abs(pool.row(p).dot(filters.row(f)))-alpha);
        }
      }
    }
    feature = feature.reshape(0,1);
    //cout << labels[f] << "," << feature << endl;
    cout << labels[f];
    for (int k=0; k<feature.cols; k++)
       cout << " " << k+1 << ":" << feature.at<double>(0,k); // data for libsvm
       //cout << "," << feature.at<double>(0,k); //data for CvMLData
    cout << endl;

  }

/*
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
 

*/



}
