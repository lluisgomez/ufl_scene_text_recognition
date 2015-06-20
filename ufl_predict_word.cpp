#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "utils.h"

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"


using namespace cv;
using namespace std;


int main(int argc, char **argv)
{

  // 1st   argument is liblinear model for text/no-text classifier
  // 2nd   argument is libsvm-scale range file for text/no-text classifier
  // 3rd   argument is liblinear model for character classifier
  // 4th   argument is libsvm-scale range file for character classifier
  // 5th   argument is 1st layer filter bank (xml file)
  // 6th   argument is an image

  cout << "running OpenCV version: "<< CV_VERSION << endl << endl;

  struct model* model_;

  if((model_=load_model(argv[1]))==0)
  {
    fprintf(stderr,"can't open model file %s\n",argv[1]);
    exit(1);
  }

  struct model* char_model_;

  if((char_model_=load_model(argv[3]))==0)
  {
    fprintf(stderr,"can't open model file %s\n",argv[3]);
    exit(1);
  }

  // Load filters bank and withenning params
  Mat filters, M, P;
  FileStorage fs(argv[5], FileStorage::READ);
  fs["D"] >> filters;
  fs["M"] >> M;
  fs["P"] >> P;
  fs.release();

  int step_size   = 4;  // sliding window step
  int window_size  = 32; // window size
  int quad_size   = 12;
  int patch_size  = 8;
  int num_quads   = 25; // extract 25 quads (12x12) from each image
  int num_tiles   = 25; // extract 25 patches (8x8) from each quad 

  double alpha    = 0.5; // used for feature representation: 
                         // scalar non-linear function z = max(0, |D*a| - alpha)


  Mat quad;
  Mat tmp;
  Mat img;
    
  Mat src  = imread(argv[6]);
  if(src.channels() != 3)
  {
      cout << "ERROR: image must be RGB" << endl;
      exit(-1);
  }
  cvtColor(src,src,COLOR_RGB2GRAY);
  resize(src,src,Size(window_size*src.cols/src.rows,window_size));

  namedWindow("image",WINDOW_NORMAL);
  //namedWindow("probabilities",WINDOW_NORMAL);

  Mat probabilities_plot = Mat::ones(100,src.cols,CV_8UC1)*255;
  double prev_prob = -1;

  // begin sliding window loop foreach detection window
  for (int x_c=0; x_c<=src.cols-window_size; x_c=x_c+step_size)
  {
    img = src(Rect(Point(x_c,0),Size(window_size,window_size)));

    int patch_count = 0;
    vector< vector<double> > data_pool(9);
 
    double t = (double)getTickCount();

    int quad_id = 1;
    for (int q_x=0; q_x<=window_size-quad_size; q_x=q_x+(quad_size/2-1))
    {
      for (int q_y=0; q_y<=window_size-quad_size; q_y=q_y+(quad_size/2-1))
      {
        Rect quad_rect = Rect(q_x,q_y,quad_size,quad_size); 
        quad = img(quad_rect);

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


    // data must be normalized within the range obtained during training
    double lower = -1.0;
    double upper =  1.0;
    Mat feature_min = Mat::zeros(1,feature.cols,CV_64FC1);
    Mat feature_max = Mat::ones(1,feature.cols,CV_64FC1);
    std::ifstream range_infile(argv[2]);
    std::string line;
    //discard first two lines
    getline(range_infile, line);
    getline(range_infile, line);
    while (getline(range_infile, line))
    {
      istringstream iss(line);
      int idx;
      double min_val, max_val;
      if (!(iss >> idx >> min_val >> max_val)) 
      {
         cout << "ERROR: reading svm-scale ranges file " << argv[2] << endl; 
         exit(0); 
      } // error

      feature_min.at<double>(0,idx-1) = min_val;
      feature_max.at<double>(0,idx-1) = max_val;
    }
    range_infile.close();

    struct feature_node *x = (struct feature_node *) 
                             malloc((feature.cols+1)*sizeof(struct feature_node));

    for (int k=0; k<feature.cols; k++)
    {
       x[k].index = k+1; // liblinear labels start at 1 not 0
       x[k].value = lower + (upper-lower) *
                      (feature.at<double>(0,k)-feature_min.at<double>(0,k))/
                      (feature_max.at<double>(0,k)-feature_min.at<double>(0,k));
    }
    x[feature.cols].index = -1;

    t = getTickCount() - t;
    //cout << " Feature extraction done in " << t/((double)getTickFrequency()*1000.) << " ms." << endl;
    t = (double)getTickCount();

    double *p = (double *) malloc(2*sizeof(double)); // 2 is number of classes
    double predict_label = predict_probability(model_,x,p);
    //fprintf(stdout,"Prediction: %g with probability %g\n",predict_label,p[0]);
    if (x_c !=0)
      cv::line(probabilities_plot,Point(x_c-step_size+(window_size/2), 100-(int)(prev_prob*100.0)),
                                  Point(x_c+(window_size/2), 100-(int)(p[0]*100.0)), Scalar(0));
    prev_prob = p[0];

    /* Now we go with the character classification*/
    range_infile.open(argv[4]);
    //discard first two lines
    getline(range_infile, line);
    getline(range_infile, line);
    while (getline(range_infile, line))
    {
      istringstream iss(line);
      int idx;
      double min_val, max_val;
      if (!(iss >> idx >> min_val >> max_val)) 
      {
         cout << "ERROR: reading svm-scale ranges file " << argv[2] << endl; 
         exit(0); 
      } // error

      feature_min.at<double>(0,idx-1) = min_val;
      feature_max.at<double>(0,idx-1) = max_val;
    }

    for (int k=0; k<feature.cols; k++)
    {
       x[k].index = k+1; // liblinear labels start at 1 not 0
       x[k].value = lower + (upper-lower) *
                      (feature.at<double>(0,k)-feature_min.at<double>(0,k))/
                      (feature_max.at<double>(0,k)-feature_min.at<double>(0,k));
    }

    double predict_char_label = predict(char_model_,x);
    //fprintf(stdout,"Prediction: %g\n",predict_label);
    string ascii = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyx0123456789";
    cout << ascii[predict_char_label-1] << " " ; cout << flush;
    t = getTickCount() - t;
    //cout << " Classification done in " << t/((double)getTickFrequency()*1000.) << " ms." << endl;
  
    free(x);
    //imshow("image",img);
    //waitKey(-1);

  }
  // end for each detection window

  vconcat(src,probabilities_plot,src);
  imwrite("tmp.jpg",src);
  imshow("image",src);
  waitKey(-1);

  free_and_destroy_model(&model_);
  return 0;
}
