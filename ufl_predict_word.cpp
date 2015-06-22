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

vector< vector<int> > generate_childs( vector<int> &segmentation, vector<int> &oversegmentation, vector<bool> &visited_nodes );
void update_beam ( int beam_size, vector< pair< double,vector<int> > > &beam, 
         vector< vector<int> > &childs, vector< vector<double> > &recognition_probabilities );
double score_segmentation( vector<int> &segmentation, string &vocabulary, vector< vector<double> > &observations, Mat transition_p );

bool beam_sort_function ( pair< double,vector<int> > i, pair< double,vector<int> > j )
{ 
  return (i.first > j.first);
}

//TODO this global vars as members of class
Mat transition_p;
string vocabulary = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyx0123456789";

int main(int argc, char **argv)
{
  // TODO do this authomatically from a lexicon!
  FileStorage fsp("/home/lluis/Escriptori/GSoC2014/opencv_contrib/modules/text/samples/OCRHMM_transitions_table.xml", FileStorage::READ);
  fsp["transition_probabilities"] >> transition_p;
  fsp.release();

  // 1st   argument is liblinear model for text/no-text classifier
  // 2nd   argument is libsvm-scale range file for text/no-text classifier
  // 3rd   argument is 1st layer filter bank (xml file)
  // 4th   argument is an image

  cout << "running OpenCV version: "<< CV_VERSION << endl << endl;

  struct model* model_;

  if((model_=load_model(argv[1]))==0)
  {
    fprintf(stderr,"can't open model file %s\n",argv[1]);
    exit(1);
  }

  // Load filters bank and withenning params
  Mat filters, M, P;
  FileStorage fs(argv[3], FileStorage::READ);
  fs["D"] >> filters;
  fs["M"] >> M;
  fs["P"] >> P;
  fs.release();

  // data must be normalized within the range obtained during training
  double lower = -1.0;
  double upper =  1.0;
  Mat feature_min = Mat::zeros(1,filters.rows*9,CV_64FC1);
  Mat feature_max = Mat::ones(1,filters.rows*9,CV_64FC1);
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
    
  Mat src  = imread(argv[4]);
  if(src.channels() != 3)
  {
      cout << "ERROR: image must be RGB" << endl;
      exit(-1);
  }
  cvtColor(src,src,COLOR_RGB2GRAY);
  resize(src,src,Size(window_size*src.cols/src.rows,window_size));

  double total_time = 0;

  int seg_points = 0;
  vector<int> oversegmentation;
  oversegmentation.push_back(seg_points);
  vector< vector<double> > recognition_probabilities;

  // begin sliding window loop foreach detection window
  for (int x_c=0; x_c<=src.cols-window_size; x_c=x_c+step_size)
  {
    double t = (double)getTickCount();

    img = src(Rect(Point(x_c,0),Size(window_size,window_size)));

    int patch_count = 0;
    vector< vector<double> > data_pool(9);
 

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

    t = (double)getTickCount() - t;
    cout << " Feature extraction done in " << t/((double)getTickFrequency()) << " s." << endl;
    total_time += t/((double)getTickFrequency());
    t = (double)getTickCount();

    //TODO use a pointer to double probabilities[model_->nr_class]; so then it can be converted into a vector<double> and use it as emission table for this position
    double probabilities[model_->nr_class];
    double *p = &probabilities[0];
    double predict_label = predict_probability(model_,x,p);
    cout << " Prediction: " << vocabulary[predict_label-1] << " with probability " << p[0] << endl; 
  
    free(x);
    t = (double)getTickCount() - t;
    cout << " Classification done in " << t/((double)getTickFrequency()) << " s." << endl;
    total_time += t/((double)getTickFrequency());

    seg_points++;
    oversegmentation.push_back(seg_points);
    vector<double> recognition_p(probabilities, probabilities+sizeof(probabilities)/sizeof(double));
    recognition_probabilities.push_back(recognition_p);

  }

  cout << "Total recognition time (s.) " << total_time << endl;
  free_and_destroy_model(&model_);


  /*Now we go here with the beam search algorithm to optimize the recognition score*/
  // TODO we need a class that takes an image, the transition and emision tables, the oversegmentation, a list of valid characters, the beam size

  cout << " we have " << oversegmentation.size() << " segmentation points." << endl;
  cout << " we have " << recognition_probabilities.size() << " recognitions." << endl;

  //convert probabilities to log probabilities
  for (int i=0; i<recognition_probabilities.size(); i++)
  {
    for (int j=0; j<recognition_probabilities[i].size(); j++)
    {
      if (recognition_probabilities[i][j] == 0)
         recognition_probabilities[i][j] = -DBL_MAX;
      else
         recognition_probabilities[i][j] = log(recognition_probabilities[i][j]);
    }
  }
  for (int i=0; i<transition_p.rows; i++)
  {
    for (int j=0; j<transition_p.cols; j++)
    {
      if (transition_p.at<double>(i,j) == 0)
         transition_p.at<double>(i,j) = -DBL_MAX;
      else
         transition_p.at<double>(i,j) = log(transition_p.at<double>(i,j));
    }
  }

  int beam_size = 50;

  //TODO this is not possible when we have a large number of possible segmentations.
  // options are using std::set<unsigned long long int> to store only the keys of visited nodes
  // but will deteriorate the time performance.
  // other ideas to discuss with Vadim.
  // it is also possible to reduce the number of seg. points in some way (e.g. use only seg.points 
  // for which there is a change on the class prediction)
  vector<bool> visited_nodes(pow(2,oversegmentation.size()),false); // hash table for visited nodes

  vector<int> start_segmentation;  
  start_segmentation.push_back(oversegmentation[0]);
  start_segmentation.push_back(oversegmentation[oversegmentation.size()-1]);
  
  vector< pair< double,vector<int> > > beam;
  beam.push_back( pair< double,vector<int> > (score_segmentation(start_segmentation, vocabulary, recognition_probabilities, transition_p), start_segmentation) );
 
  vector< vector<int> > childs = generate_childs(start_segmentation,oversegmentation, visited_nodes);
  if (!childs.empty())
    update_beam(beam_size, beam, childs, recognition_probabilities);
  cout << "beam size " << beam.size() << " best score " << beam[0].first<< endl;
 
  int generated_chids = childs.size(); 
  while (generated_chids != 0)
  {
    generated_chids = 0;
    vector< pair< double,vector<int> > > old_beam = beam;
    
    for (int i=0; i<old_beam.size(); i++)
    {
      childs = generate_childs(old_beam[i].second,oversegmentation, visited_nodes);
      if (!childs.empty()) 
        update_beam(beam_size, beam, childs, recognition_probabilities);
      generated_chids += childs.size();
    }
    cout << "beam size " << beam.size() << " best score " << beam[0].first << endl;
  }

  
  cout << "FINISHED ! Best score found : " << endl;
  score_segmentation(beam[0].second, vocabulary, recognition_probabilities, transition_p);

  // Release mem of global Mat // TODO this is not necessary if Mat is member of class
  transition_p.release();

  return 0;
}


////////////////////////////////////////////////////////////

// TODO the way we expand nodes makes the recognition score heuristic not monotonic
// it should start from left node 0 and grow always to the right.

vector< vector<int> > generate_childs(vector<int> &segmentation, vector<int> &oversegmentation, vector<bool> &visited_nodes)
{
  cout << " generate childs  for [";
  for (int i = 0 ; i < segmentation .size(); i++)
      cout << segmentation[i] << ",";
  cout << "] ";
  vector< vector<int> > childs;
  for (int i=0; i<oversegmentation.size(); i++)
  {
    int seg_point = oversegmentation[i];
    if (find(segmentation.begin(), segmentation.end(), seg_point) == segmentation.end())
    {
      cout << seg_point << " " ;
      vector<int> child = segmentation;
      child.push_back(seg_point);
      sort(child.begin(), child.end());
      int key = 0;
      for (int j=0; j<child.size(); j++)
      {
key += pow(2,oversegmentation.size()-(oversegmentation.end()-find(oversegmentation.begin(), oversegmentation.end(), child[j])));
      }
      if (!visited_nodes[key])
      {
        childs.push_back(child);
        visited_nodes[key] = true;
      }
    }
  }
  cout << endl;
  return childs;
}


////////////////////////////////////////////////////////////

void update_beam (int beam_size, vector< pair< double,vector<int> > > &beam, vector< vector<int> > &childs, vector< vector<double> > &recognition_probabilities)
{
  double min_score = -DBL_MAX; //min score value to be part of the beam
  if (beam.size() == beam_size)
    min_score = beam[beam.size()-1].first; //last element has the lowest score
  for (int i=0; i<childs.size(); i++)
  {
    double score = score_segmentation(childs[i], vocabulary, recognition_probabilities, transition_p);
    if (score > min_score)
    {
      beam.push_back(pair< double,vector<int> >(score,childs[i]));
      sort(beam.begin(),beam.end(),beam_sort_function);
      if (beam.size() > beam_size)
      {
        beam.pop_back();
        min_score = beam[beam.size()-1].first;
      }
    }
  }
}


////////////////////////////////////////////////////////////
// TODO Add heuristics to the score function (see PhotoOCR paper)
// e.g.: in some cases we discard a segmentation because it includes a very large character
//       in other cases we do it because the overlapping between two chars is too large
//       etc.
double score_segmentation(vector<int> &segmentation, string &vocabulary, vector< vector<double> > &observations, Mat transition_p)
{

  //TODO This must be extracted from dictionary
  vector<double> start_p(vocabulary.size());
  for (int i=0; i<(int)vocabulary.size(); i++)
      start_p[i] = log(1.0/vocabulary.size());


  Mat V = Mat::ones((int)segmentation.size()-1,(int)vocabulary.size(),CV_64FC1);
  V = V * -DBL_MAX;
  vector<string> path(vocabulary.size());

  // Initialize base cases (t == 0)
  for (int i=0; i<(int)vocabulary.size(); i++)
  {
      V.at<double>(0,i) = start_p[i] + observations[segmentation[1]-1][i];
      //cout << " setting V.at<double>("<<0<<","<< i << ")" << "= " << V.at<double>(0,i) << endl;
      path[i] = vocabulary.at(i);
  }


  // Run Viterbi for t > 0
  for (int t=1; t<(int)segmentation.size()-1; t++)
  {

      vector<string> newpath(vocabulary.size());

      for (int i=0; i<(int)vocabulary.size(); i++)
      {
          double max_prob = -DBL_MAX;
          int best_idx = 0;
          for (int j=0; j<(int)vocabulary.size(); j++)
          {
              double prob = V.at<double>(t-1,j) + transition_p.at<double>(j,i) + observations[segmentation[t+1]-1][i];
              if ( prob > max_prob)
              {
                  max_prob = prob;
                  best_idx = j;
              }
          }

          //cout << " setting V.at<double>("<<t<<","<< i << ")" << "= " << max_prob << endl;
          V.at<double>(t,i) = max_prob;
          newpath[i] = path[best_idx] + vocabulary.at(i);
      }

      // Don't need to remember the old paths
      path.swap(newpath);
  }

  double max_prob = -DBL_MAX;
  int best_idx = 0;
  for (int i=0; i<(int)vocabulary.size(); i++)
  {
      double prob = V.at<double>((int)segmentation.size()-2,i);
      //cout << " getting V.at<double>("<<(int)segmentation.size()-2<<","<< i << ")" << "= " << prob << endl;
      if ( prob > max_prob)
      {
          max_prob = prob;
          best_idx = i;
      }
  }

  //cout << path[best_idx] << endl;
  cout << " score " << max_prob / (segmentation.size()-1) << " " << path[best_idx] << endl;
  return max_prob / (segmentation.size()-1);
}
