#include <opencv2/opencv.hpp>
#include <utility>      // std::pair, std::make_pair
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


vector< vector<int> > generate_childs( vector<int> &segmentation, vector<int> &oversegmentation );
void update_beam ( int beam_size, vector< pair< float,vector<int> > > &beam, 
                   vector< vector<int> > &childs );
float score_segmentation( vector<int> &segmentation );

bool beam_sort_function ( pair< float,vector<int> > i, pair< float,vector<int> > j )
{ 
  return (i.first > j.first);
}


// TODO this globals must be members of the class
Mat image; // an image (32px. height) corresponds to a word
int myints[] = {0,16,17,23,32,36,43,56,64,67,69,78,82,92,96,102,109,112,128};
vector<int> oversegmentation (myints, myints + sizeof(myints) / sizeof(int)); // a list of pixels, i.e. possible seg. points.
vector<bool> visited_nodes(pow(2,oversegmentation.size()),false); // hash table for visited nodes


int beam_size = 50;

int main(int argc, char **argv)
{
  vector<int> start_segmentation;  
  start_segmentation.push_back(oversegmentation[0]);
  start_segmentation.push_back(oversegmentation[oversegmentation.size()-1]);
  
  vector< pair< float,vector<int> > > beam;
  beam.push_back( pair< float,vector<int> > (score_segmentation(start_segmentation), 
                                             start_segmentation) );
 
  vector< vector<int> > childs = generate_childs(start_segmentation,oversegmentation);
  if (!childs.empty())
    update_beam(beam_size, beam, childs);
  cout << "beam size " << beam.size() << " best score " << beam[0].first<< endl;
 
  int generated_chids = childs.size(); 
  while (generated_chids != 0)
  {
    generated_chids = 0;
    vector< pair< float,vector<int> > > old_beam = beam;
    
    for (int i=0; i<old_beam.size(); i++)
    {
      childs = generate_childs(old_beam[i].second,oversegmentation);
      if (!childs.empty()) 
        update_beam(beam_size, beam, childs);
      generated_chids += childs.size();
    }
    cout << "beam size " << beam.size() << " best score " << beam[0].first<< endl;
    //char string [256];gets (string);
  }

  return 0;
}


////////////////////////////////////////////////////////////

vector< vector<int> > generate_childs(vector<int> &segmentation, vector<int> &oversegmentation)
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

void update_beam (int beam_size, vector< pair< float,vector<int> > > &beam, vector< vector<int> > &childs)
{
  float min_score = -1; //min score value to be part of the beam
  if (beam.size() == beam_size)
    min_score = beam[beam.size()-1].first; //last element has the lowest score
  for (int i=0; i<childs.size(); i++)
  {
    float score = score_segmentation(childs[i]);
    if (score > min_score)
    {
      beam.push_back(pair< float,vector<int> >(score,childs[i]));
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

float score_segmentation(vector<int> &segmentation)
{
   //for the moment just return a random float
   // in future it should return the score using the recognition confidences and lang model
   cout << "score" << endl;
   return (float)rand()/RAND_MAX;
}
